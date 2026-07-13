"""M0 scaffold sanity tests: the package imports and shared utilities work."""

import json

import inference_lab
from inference_lab.common import EndpointConfig, get_logger, log_event


def test_package_imports():
    assert inference_lab.__version__


def test_endpoint_config_defaults():
    cfg = EndpointConfig(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct")
    assert cfg.api_key == "EMPTY"
    assert cfg.timeout_s > 0


def test_endpoint_config_from_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_LAB_BASE_URL", "http://pod:8000/v1")
    monkeypatch.setenv("INFERENCE_LAB_MODEL", "test-model")
    cfg = EndpointConfig.from_env()
    assert cfg.base_url == "http://pod:8000/v1"
    assert cfg.model == "test-model"


def test_log_event_appends_jsonl(tmp_path):
    path = tmp_path / "runs" / "events.jsonl"
    log_event(path, {"kind": "request", "ttft_ms": 123.4})
    log_event(path, {"kind": "request", "ttft_ms": 99.9, "ts": 1.0})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["ttft_ms"] == 123.4
    assert "ts" in lines[0]
    assert lines[1]["ts"] == 1.0


def test_get_logger_is_namespaced():
    assert get_logger("loadtest").name == "inference_lab.loadtest"
