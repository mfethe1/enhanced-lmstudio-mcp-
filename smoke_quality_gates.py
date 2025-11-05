#!/usr/bin/env python3
"""Smoke test for Quality Gates tools presence in MCP server."""

import sys
sys.path.insert(0, '.')

from server import get_all_tools

def main():
    data = get_all_tools()
    tools = data.get('tools', []) if isinstance(data, dict) else data
    names = [t.get('name') for t in tools if isinstance(t, dict)]
    needed = {
        'evaluate_quality',
        'enforce_quality_gate',
        'get_quality_stats',
    }
    missing = [n for n in needed if n not in names]
    print(f"Found {len(names)} tools; checking quality gate tools...")
    for n in needed:
        print(f" - {n}: {'OK' if n in names else 'MISSING'}")
    if missing:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
