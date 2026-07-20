#!/usr/bin/env python3
"""Summarize a gateway JSONL request log: cost, latency, and routing decisions.

Thin wrapper so the aggregation lives in the package (and under test):
see ``inference_lab.gateway.report``.

Usage: ``python scripts/gateway_report.py <gateway_log.jsonl>``
"""

from inference_lab.gateway.report import main

if __name__ == "__main__":
    main()
