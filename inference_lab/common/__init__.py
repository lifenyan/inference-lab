"""Shared utilities: endpoint/model configuration and structured JSONL logging."""

from inference_lab.common.config import EndpointConfig
from inference_lab.common.logging import get_logger, log_event

__all__ = ["EndpointConfig", "get_logger", "log_event"]
