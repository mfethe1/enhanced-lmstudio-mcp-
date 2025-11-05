import os
import sys
import json
import argparse
import requests
from pathlib import Path

def parse_env_file(path: Path) -> dict:
    pairs = {}
    if not path.exists():
        return pairs
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("export "): s = s[len("export "):].strip()
        if "=" not in s: continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        pairs[k] = v
    return pairs

def mask(v: str) -> str:
    if not v: return "<empty>"
    return v[:4] + "..." + v[-4:]

def main():
    ap = argparse.ArgumentParser(description="Standalone Anthropic 'hi' probe using HTTP")
    ap.add_argument("--env", default=".secrets/.env.local", help="Path to env file to read (default: .secrets/.env.local)")
    ap.add_argument("--model", default=None, help="Override model (default from env file or claude-3-5-sonnet-latest)")
    ap.add_argument("--base", default=None, help="Base URL (default from env file or https://api.anthropic.com/v1)")
    ap.add_argument("--version", default=None, help="Anthropic version header (default from env or 2023-06-01)")
    ap.add_argument("--say", default="hi", help="Message to send (default: hi)")
    args = ap.parse_args()

    env_pairs = parse_env_file(Path(args.env))

    api_key = env_pairs.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    model = args.model or env_pairs.get("ANTHROPIC_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
    base = args.base or env_pairs.get("ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
    version = args.version or env_pairs.get("ANTHROPIC_VERSION") or os.getenv("ANTHROPIC_VERSION") or "2023-06-01"

    print(f"Using key: {mask(api_key)}; model: {model}; base: {base}; version: {version}")

    if not api_key:
        print("ERROR: No ANTHROPIC_API_KEY found in env file or environment.")
        sys.exit(2)

    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": args.say}],
    }

    url = base.rstrip("/") + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": version,
        "content-type": "application/json",
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Raw: {r.text[:500]}")
        r.raise_for_status()
        data = r.json()
        parts = data.get("content", [])
        txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        print(f"Reply: {txt!r}")
        sys.exit(0)
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

