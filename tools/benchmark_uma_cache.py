#!/usr/bin/env python3
"""Measure cold versus warm UMA service calls without loading a second model."""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        base.rstrip("/") + path,
        data=body,
        method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def wait_for_job(base: str, job_id: str) -> tuple[dict, float]:
    started = time.perf_counter()
    while True:
        job = request(base, f"/api/v1/jobs/{job_id}")
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job, time.perf_counter() - started
        time.sleep(0.25)


def run_one(base: str, payload: dict) -> dict:
    before = request(base, "/api/v1/health")
    wall_started = time.perf_counter()
    submitted = request(base, "/api/v1/jobs", payload)
    job, poll_seconds = wait_for_job(base, submitted["id"])
    wall_seconds = time.perf_counter() - wall_started
    result = job.get("result") or {}
    return {
        "job_id": submitted["id"],
        "status": job["status"],
        "wall_seconds": round(wall_seconds, 3),
        "poll_seconds": round(poll_seconds, 3),
        "calculation_seconds": result.get("elapsed_seconds"),
        "worker_pid_before": before.get("worker_pid"),
        "worker_pid_after": request(base, "/api/v1/health").get("worker_pid"),
        "error": job.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    payload = {
        "model_name": "uma-s-1p2p1",
        "task_name": "omol",
        "calculation": "single_point",
        "structure": {
            "atoms": [
                {"symbol": "O", "position": [0.0, 0.0, 0.0]},
                {"symbol": "H", "position": [0.757, 0.586, 0.0]},
                {"symbol": "H", "position": [-0.757, 0.586, 0.0]},
            ],
            "charge": 0,
            "spin": 1,
            "name": "uma-cache-benchmark-water",
        },
    }
    try:
        health = request(args.url, "/api/v1/health")
        if not health.get("worker_enabled"):
            raise RuntimeError("UMA Worker is disabled")
        cold = run_one(args.url, payload)
        warm = run_one(args.url, payload)
    except (HTTPError, URLError, RuntimeError) as exc:
        raise SystemExit(f"benchmark failed: {exc}") from exc
    print(json.dumps({"cold": cold, "warm": warm}, indent=2))


if __name__ == "__main__":
    main()
