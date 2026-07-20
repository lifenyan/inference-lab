#!/usr/bin/env python3
"""Drive the M8 end-to-end demo workload through the gateway.

Sends a mixed workload that exercises every routing rule; the gateway's JSONL
log is the artifact (summarize it with ``scripts/gateway_report.py``), this
script just prints what each response's ``x-gateway-*`` headers say happened.

Stages (``--stage``):

- ``main``: short chat requests (local), a shared-system-prefix batch (local —
  watch TTFT collapse after the first request populates the prefix cache),
  unique long-context requests (over the token threshold → fallback), and a
  request for a model not served locally (→ fallback).
- ``failure``: run AFTER stopping vLLM on the pod — shows per-request fallback
  (``local_error_fallback``) and then the circuit opening (``circuit_open``).

Usage:
    python scripts/run_m8_demo.py --gateway http://127.0.0.1:8080 --stage main
    # ...stop vLLM on the pod...
    python scripts/run_m8_demo.py --gateway http://127.0.0.1:8080 --stage failure
"""

import argparse
import json
import time

import httpx

# ~1,200 estimated tokens (4,800 chars / 4): a realistic RAG/agent system prompt,
# deliberately UNDER the routing threshold — shared-prefix traffic is the cheapest
# to serve locally (M6: 86% cache hit), and the demo should show it staying local.
SHARED_SYSTEM_PREFIX = (
    "You are the support assistant for the Aurora Analytics platform. "
    "Follow these policies exactly. "
    + " ".join(
        f"Policy {i}: for questions about topic {i}, first consult the knowledge base, "
        "cite the relevant article id, keep answers under three sentences, escalate "
        "billing disputes to a human agent, and never disclose internal tooling names."
        for i in range(1, 24)
    )
)

QUESTIONS = [
    "How do I reset my password?",
    "Can I export my dashboard as a PDF?",
    "Why is my data sync delayed?",
    "How do I add a teammate to my workspace?",
    "What does the 'stale snapshot' warning mean?",
    "How do I change my billing email?",
    "Can I schedule a report to send weekly?",
    "How do I revoke an API key?",
]

# ~2,000 estimated tokens of *unique* content per request: over the threshold,
# and exactly the traffic the threshold exists for (KV-wall risk, report §8).
def unique_long_prompt(i: int) -> str:
    return (
        f"Document {i}. " + " ".join(
            f"Section {j}: finding {i}-{j} indicates the metric moved by {j % 17} points "
            "during the observation window, which the appendix attributes to seasonal load."
            for j in range(60)
        )
        + " Summarize the three most important findings in this document."
    )


def send(client: httpx.Client, gateway: str, body: dict, label: str) -> None:
    """Send one request (streaming if body says so) and print the routing outcome."""
    t0 = time.perf_counter()
    ttft = None
    try:
        if body.get("stream"):
            with client.stream(
                "POST", f"{gateway}/v1/chat/completions", json=body, timeout=180.0
            ) as response:
                headers = response.headers
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices") or []
                    if ttft is None and choices and (choices[0].get("delta") or {}).get("content"):
                        ttft = time.perf_counter() - t0
                status = response.status_code
        else:
            response = client.post(f"{gateway}/v1/chat/completions", json=body, timeout=180.0)
            headers, status = response.headers, response.status_code
    except httpx.HTTPError as exc:
        print(f"  {label:<34} ERROR {type(exc).__name__}: {exc}")
        return
    latency = time.perf_counter() - t0
    print(
        f"  {label:<34} {status} {headers.get('x-gateway-backend', '?'):<9}"
        f" {headers.get('x-gateway-route-reason', '?'):<22}"
        f" latency {latency:6.2f}s" + (f"  ttft {ttft:5.2f}s" if ttft is not None else "")
    )


def stage_main(client: httpx.Client, gateway: str, model: str) -> None:
    print("— short chat requests (expect: local / default_local)")
    for i, question in enumerate(QUESTIONS[:3]):
        send(
            client, gateway,
            {"model": model, "messages": [{"role": "user", "content": question}],
             "max_tokens": 96, "stream": i % 2 == 0},
            f"chat[{i}] {'stream' if i % 2 == 0 else 'json'}",
        )

    print("— shared-system-prefix batch (expect: local; TTFT drops after request 0")
    print("  as the ~1.2k-token prefix lands in the prefix cache)")
    for i, question in enumerate(QUESTIONS):
        send(
            client, gateway,
            {"model": model, "max_tokens": 96, "stream": True,
             "messages": [
                 {"role": "system", "content": SHARED_SYSTEM_PREFIX},
                 {"role": "user", "content": question},
             ]},
            f"prefix[{i}] stream",
        )

    print("— unique long-context requests (expect: fallback / over_token_threshold)")
    for i in range(2):
        send(
            client, gateway,
            {"model": model, "max_tokens": 96,
             "messages": [{"role": "user", "content": unique_long_prompt(i)}]},
            f"long[{i}] json",
        )

    print("— model not served locally (expect: fallback / model_not_local)")
    send(
        client, gateway,
        {"model": "gpt-5.4-nano", "max_tokens": 64,
         "messages": [{"role": "user", "content": "In one sentence: what is a KV cache?"}]},
        "other-model json",
    )


def stage_failure(client: httpx.Client, gateway: str, model: str) -> None:
    print("— local backend is down (expect: local_error_fallback until the")
    print("  circuit opens, then circuit_open — no more local connect attempts)")
    for i in range(5):
        send(
            client, gateway,
            {"model": model, "max_tokens": 64,
             "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]},
            f"failure[{i}] json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 end-to-end demo workload")
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                        help="the locally served model name")
    parser.add_argument("--stage", choices=["main", "failure"], default="main")
    args = parser.parse_args()

    with httpx.Client() as client:
        if args.stage == "main":
            stage_main(client, args.gateway, args.model)
        else:
            stage_failure(client, args.gateway, args.model)


if __name__ == "__main__":
    main()
