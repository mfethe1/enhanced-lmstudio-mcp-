#!/usr/bin/env python3
"""Check for duplicate tool names in the MCP server."""

import sys
sys.path.insert(0, '.')

from server import get_all_tools

def check_duplicates():
    tools = get_all_tools()
    # Tools might be a list of Tool objects or dict
    if isinstance(tools, list) and len(tools) > 0:
        if hasattr(tools[0], 'name'):
            names = [t.name for t in tools]
        elif isinstance(tools[0], dict):
            names = [t['name'] for t in tools]
        else:
            names = tools  # Already a list of names
    else:
        print(f"Unexpected tools format: {type(tools)}")
        return False
    
    # Find duplicates
    duplicates = {}
    for name in names:
        if names.count(name) > 1 and name not in duplicates:
            duplicates[name] = names.count(name)
    
    print(f"Total tools: {len(tools)}")
    print(f"Unique names: {len(set(names))}")
    
    if duplicates:
        print(f"\n⚠️ Found {len(duplicates)} duplicate tool names:")
        for name, count in duplicates.items():
            print(f"  - {name}: appears {count} times")
    else:
        print("✅ No duplicate tool names found")
    
    return len(duplicates) == 0

if __name__ == "__main__":
    success = check_duplicates()
    sys.exit(0 if success else 1)