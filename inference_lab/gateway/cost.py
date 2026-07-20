"""Per-request cost computation.

Local cost comes from the config's single stated $/1M-output figure and is
always paired with that figure's assumption string — it is utilization- and
regime-dependent, not a constant of nature (report §7). Fallback cost uses
exact token counts against list prices: ``in × in_price + out × out_price``.
The report's 2.83:1 output-equivalent conversion is deliberately NOT used
here — it was a bridge for comparing shape-level aggregates, wrong where
exact per-request token counts exist.
"""

from inference_lab.gateway.config import GatewayConfig
from inference_lab.gateway.routing import LOCAL


def request_cost(
    config: GatewayConfig,
    backend: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> tuple[float | None, str]:
    """Return ``(cost_usd, basis)`` for one served request.

    ``basis`` states the price source and its assumptions; it is logged with
    every request so aggregated totals can restate what they rest on. Cost is
    None when the backend reported no usable token counts (failed requests).
    """
    if backend == LOCAL:
        local = config.local
        basis = (
            f"local ${local.usd_per_1m_output_tokens}/1M output tokens"
            f" ({local.cost_assumption})"
        )
        if output_tokens is None:
            return None, basis
        return output_tokens * local.usd_per_1m_output_tokens / 1e6, basis

    fb = config.fallback
    basis = (
        f"fallback {fb.model} list prices ${fb.usd_per_1m_input_tokens}/1M in"
        f" + ${fb.usd_per_1m_output_tokens}/1M out (fetched {fb.prices_fetched_on})"
    )
    if input_tokens is None or output_tokens is None:
        return None, basis
    cost = (
        input_tokens * fb.usd_per_1m_input_tokens / 1e6
        + output_tokens * fb.usd_per_1m_output_tokens / 1e6
    )
    return cost, basis
