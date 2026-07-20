"""Unit tests: routing policy, circuit breaker, token estimation, cost computation."""

import time

from inference_lab.gateway.config import (
    FallbackBackendConfig,
    GatewayConfig,
    LocalBackendConfig,
    RoutingConfig,
)
from inference_lab.gateway.cost import request_cost
from inference_lab.gateway.routing import (
    FALLBACK,
    LOCAL,
    REASON_CIRCUIT_OPEN,
    REASON_DEFAULT,
    REASON_LOCAL_UNHEALTHY,
    REASON_MODEL_NOT_LOCAL,
    REASON_OVER_THRESHOLD,
    CircuitBreaker,
    HealthMonitor,
    Router,
    estimate_prompt_tokens,
    health_url,
)


def make_config(**routing_overrides) -> GatewayConfig:
    routing = {
        "prompt_token_threshold": 100,
        "health_check_interval_s": 5.0,
        "circuit_breaker_failures": 3,
        "circuit_breaker_open_s": 30.0,
    } | routing_overrides
    return GatewayConfig(
        local=LocalBackendConfig(
            base_url="http://local:8000/v1",
            served_models=["local-model"],
            usd_per_1m_output_tokens=100.0,
            cost_assumption="test assumption: full utilization, fresh traffic",
        ),
        fallback=FallbackBackendConfig(
            base_url="http://fallback:9000/v1",
            model="fallback-model",
            api_key_env="GW_TEST_FALLBACK_KEY",
            usd_per_1m_input_tokens=10.0,
            usd_per_1m_output_tokens=20.0,
            prices_fetched_on="2026-07-19",
        ),
        routing=RoutingConfig(**routing),
        log_path="unused.jsonl",
    )


class TestEstimatePromptTokens:
    def test_empty_messages(self):
        assert estimate_prompt_tokens([]) == 0

    def test_scales_with_content_length(self):
        # 400 chars at ~4 chars/token + 4 framing tokens.
        est = estimate_prompt_tokens([{"role": "user", "content": "x" * 400}])
        assert est == 104

    def test_multiple_messages_add_framing(self):
        messages = [{"role": "user", "content": "abcd"}] * 3
        assert estimate_prompt_tokens(messages) == 3 * (4 + 1)

    def test_multimodal_content_parts(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "x" * 40}]}]
        assert estimate_prompt_tokens(messages) == 4 + 10

    def test_null_content(self):
        assert estimate_prompt_tokens([{"role": "assistant", "content": None}]) == 4


def test_health_url_strips_v1_suffix():
    assert health_url("http://pod:8000/v1") == "http://pod:8000/health"
    assert health_url("http://pod:8000/v1/") == "http://pod:8000/health"
    assert health_url("http://pod:8000") == "http://pod:8000/health"


class TestCircuitBreaker:
    def test_stays_closed_below_threshold(self):
        breaker = CircuitBreaker(failure_threshold=2, open_s=30.0)
        breaker.record_failure()
        assert not breaker.is_open

    def test_opens_at_threshold_and_recloses_after_window(self):
        breaker = CircuitBreaker(failure_threshold=2, open_s=0.1)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open
        time.sleep(0.15)
        assert not breaker.is_open  # half-open: next request is the trial

    def test_failure_after_window_reopens_immediately(self):
        breaker = CircuitBreaker(failure_threshold=2, open_s=0.1)
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.15)
        breaker.record_failure()  # trial fails: count is still at/above threshold
        assert breaker.is_open

    def test_success_resets(self):
        breaker = CircuitBreaker(failure_threshold=2, open_s=0.1)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open


class TestRouterPrecedence:
    def _router(self, threshold: int = 100) -> tuple[Router, CircuitBreaker, HealthMonitor]:
        config = make_config(prompt_token_threshold=threshold)
        breaker = CircuitBreaker(1, 30.0)
        monitor = HealthMonitor("http://local:8000/health", 5.0)
        return Router(config.routing, ["local-model"], breaker, monitor), breaker, monitor

    def test_default_is_local(self):
        router, _, _ = self._router()
        decision = router.decide("local-model", 10)
        assert (decision.backend, decision.reason) == (LOCAL, REASON_DEFAULT)

    def test_unknown_model_goes_to_fallback(self):
        router, _, _ = self._router()
        decision = router.decide("gpt-other", 10)
        assert (decision.backend, decision.reason) == (FALLBACK, REASON_MODEL_NOT_LOCAL)

    def test_over_threshold_goes_to_fallback(self):
        router, _, _ = self._router(threshold=100)
        assert router.decide("local-model", 100).backend == LOCAL  # boundary: == is under
        decision = router.decide("local-model", 101)
        assert (decision.backend, decision.reason) == (FALLBACK, REASON_OVER_THRESHOLD)

    def test_open_circuit_beats_threshold(self):
        router, breaker, _ = self._router()
        breaker.record_failure()
        decision = router.decide("local-model", 101)
        assert (decision.backend, decision.reason) == (FALLBACK, REASON_CIRCUIT_OPEN)

    def test_unhealthy_local_goes_to_fallback(self):
        router, _, monitor = self._router()
        monitor.healthy = False
        decision = router.decide("local-model", 10)
        assert (decision.backend, decision.reason) == (FALLBACK, REASON_LOCAL_UNHEALTHY)

    def test_unknown_model_beats_open_circuit(self):
        router, breaker, _ = self._router()
        breaker.record_failure()
        assert router.decide("gpt-other", 10).reason == REASON_MODEL_NOT_LOCAL


class TestPrepareFallbackPayload:
    def _adapt(self, payload: dict) -> dict:
        from inference_lab.gateway.app import prepare_fallback_payload

        return prepare_fallback_payload(payload, "fallback-model", ["local-model"])

    def test_local_model_is_rewritten_other_models_pass_through(self):
        assert self._adapt({"model": "local-model"})["model"] == "fallback-model"
        assert self._adapt({"model": "gpt-other"})["model"] == "gpt-other"

    def test_max_tokens_renamed_for_openai_newer_models(self):
        adapted = self._adapt({"model": "local-model", "max_tokens": 96})
        assert "max_tokens" not in adapted
        assert adapted["max_completion_tokens"] == 96

    def test_existing_max_completion_tokens_wins(self):
        adapted = self._adapt({"max_tokens": 96, "max_completion_tokens": 32})
        assert adapted["max_completion_tokens"] == 32
        assert "max_tokens" not in adapted

    def test_vllm_only_ignore_eos_dropped_and_original_untouched(self):
        original = {"model": "local-model", "ignore_eos": True, "max_tokens": 8}
        adapted = self._adapt(original)
        assert "ignore_eos" not in adapted
        assert original == {"model": "local-model", "ignore_eos": True, "max_tokens": 8}


class TestRequestCost:
    def test_local_cost_is_output_tokens_times_rate(self):
        cost, basis = request_cost(make_config(), LOCAL, input_tokens=500, output_tokens=200)
        assert cost is not None and abs(cost - 200 * 100.0 / 1e6) < 1e-12
        assert "test assumption" in basis  # the assumption travels with every cost

    def test_fallback_cost_uses_exact_in_and_out_tokens(self):
        cost, basis = request_cost(make_config(), FALLBACK, input_tokens=500, output_tokens=200)
        expected = 500 * 10.0 / 1e6 + 200 * 20.0 / 1e6
        assert cost is not None and abs(cost - expected) < 1e-12
        assert "fetched 2026-07-19" in basis  # prices carry their date

    def test_missing_usage_yields_no_cost_but_keeps_basis(self):
        cost, basis = request_cost(make_config(), LOCAL, input_tokens=None, output_tokens=None)
        assert cost is None and "test assumption" in basis
        cost, _ = request_cost(make_config(), FALLBACK, input_tokens=10, output_tokens=None)
        assert cost is None
