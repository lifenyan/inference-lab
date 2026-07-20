"""OpenAI-compatible gateway: routing between the self-hosted vLLM backend and
a commercial-API fallback, with per-request cost/latency logging (M8).

Entry points: ``create_app`` (app factory), ``python -m inference_lab.gateway``
(server), ``inference_lab.gateway.report`` / ``scripts/gateway_report.py``
(log summarizer).
"""

from inference_lab.gateway.config import GatewayConfig

__all__ = ["GatewayConfig"]
