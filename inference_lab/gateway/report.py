"""Aggregate a gateway JSONL request log into a cost/latency/routing summary.

Cost totals are grouped by their logged ``cost_basis`` so every dollar figure
appears next to the assumption it rests on: local $/token is utilization- and
regime-dependent (report §7 — ~25× spread between c=1 and c=64 steady-state,
1.65× between fresh and cache-warm traffic), and fallback prices are dated
list prices. A cost total without its basis is not a number worth quoting.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference_lab.loadtest.stats import percentile


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Read the request events out of one JSONL gateway log."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "request":
            events.append(event)
    return events


def _p50_p99(values: list[float]) -> str:
    if not values:
        return "     —     "
    return f"{percentile(values, 50):6.2f} / {percentile(values, 99):6.2f}"


def summarize(events: list[dict[str, Any]]) -> str:
    """Render the human-readable summary table for a list of request events."""
    ok = [e for e in events if e.get("status") == "ok"]
    errors = len(events) - len(ok)
    lines = [f"Gateway request log — {len(events)} requests ({len(ok)} ok, {errors} errors)", ""]

    lines.append("Per backend:")
    header = (
        f"  {'backend':<10} {'req':>4} {'err':>4} {'tokens in':>10} {'tokens out':>10}"
        f" {'cost USD':>10}   {'latency p50/p99 (s)':^19}   {'TTFT p50/p99 (s)':^15}"
    )
    lines.append(header)
    for backend in ("local", "fallback"):
        group = [e for e in events if e.get("backend") == backend]
        if not group:
            continue
        group_ok = [e for e in group if e.get("status") == "ok"]
        tokens_in = sum(e.get("input_tokens") or 0 for e in group)
        tokens_out = sum(e.get("output_tokens") or 0 for e in group)
        cost = sum(e.get("cost_usd") or 0.0 for e in group)
        latencies = [e["latency_s"] for e in group_ok if e.get("latency_s") is not None]
        ttfts = [e["ttft_s"] for e in group_ok if e.get("ttft_s") is not None]
        lines.append(
            f"  {backend:<10} {len(group):>4} {len(group) - len(group_ok):>4}"
            f" {tokens_in:>10,} {tokens_out:>10,} {cost:>10.6f}"
            f"   {_p50_p99(latencies):^19}   {_p50_p99(ttfts):^15}"
        )
    lines.append("")

    lines.append("Routing reasons:")
    reasons = Counter(e.get("route_reason", "unknown") for e in events)
    for reason, count in reasons.most_common():
        lines.append(f"  {reason:<24} {count:>4}")
    lines.append("")

    lines.append("Cost totals by pricing basis (a cost is only meaningful with its assumption):")
    by_basis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("cost_basis"):
            by_basis[event["cost_basis"]].append(event)
    for basis, group in sorted(by_basis.items()):
        cost = sum(e.get("cost_usd") or 0.0 for e in group)
        lines.append(f"  ${cost:.6f} over {len(group)} requests — {basis}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a gateway JSONL request log (cost / latency / routing)."
    )
    parser.add_argument("log", type=Path, help="path to the gateway's JSONL log file")
    args = parser.parse_args(argv)
    events = load_requests(args.log)
    if not events:
        raise SystemExit(f"no request events found in {args.log}")
    print(summarize(events))


if __name__ == "__main__":
    main()
