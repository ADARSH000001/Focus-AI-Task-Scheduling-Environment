#!/usr/bin/env python3
"""Simple remote smoke test for a deployed Hugging Face Space."""

from __future__ import annotations

import argparse
import json
import sys
from urllib import request


def get_json(url: str) -> tuple[int, str]:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def post_json(url: str, payload: dict) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    checks = {}
    for path in ("/", "/health", "/validate"):
        status, body = get_json(base_url + path)
        checks[path] = {"status": status, "body": body[:300]}

    status, body = post_json(base_url + "/reset", {"difficulty": "easy"})
    checks["/reset"] = {"status": status, "body": body[:300]}

    status, body = post_json(base_url + "/step", {"action": "start_task('report')"})
    checks["/step"] = {"status": status, "body": body[:300]}

    print(json.dumps(checks, indent=2))
    if any(item["status"] != 200 for item in checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
