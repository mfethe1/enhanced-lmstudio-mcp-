import re
import json

# Read the server.py file
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the get_all_tools function
pattern = r'def get_all_tools\(\):(.*?)^def '
match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

if match:
    func_content = match.group(1)
    
    # Extract all tool definitions
    tool_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"[^}]*?\}'
    tools = re.findall(tool_pattern, func_content, re.DOTALL)
    
    # Find duplicates
    from collections import Counter
    tool_counts = Counter(tools)
    duplicates = {name: count for name, count in tool_counts.items() if count > 1}
    
    if duplicates:
        print("Found duplicate tool definitions in get_all_tools:")
        for name, count in duplicates.items():
            print(f"  {name}: {count} times")
    else:
        print("No duplicate tool definitions in get_all_tools")
else:
    print("Could not find get_all_tools function")