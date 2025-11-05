import json
import re
from collections import Counter

# Read the server.py file
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all tool name definitions
tool_pattern = r'"name": "([^"]+)"'
matches = re.findall(tool_pattern, content)

# Count occurrences
tool_counts = Counter(matches)

# Find duplicates
duplicates = {name: count for name, count in tool_counts.items() if count > 1}

if duplicates:
    print('Found duplicate tool names:')
    for name, count in sorted(duplicates.items()):
        print(f'  {name}: {count} occurrences')
else:
    print('No duplicate tool names found')

# Show all unique tools
print(f'\nTotal unique tools: {len(tool_counts)}')
print('\nAll tool names:')
for name in sorted(tool_counts.keys()):
    print(f'  - {name}')

# Check for common tool names that might conflict with other MCP servers
common_names = [
    'execute_code', 'read_file', 'write_file', 'list_directory',
    'search', 'analyze', 'test', 'run', 'get_status', 'help',
    'debug', 'compile', 'build', 'deploy', 'format', 'lint'
]

print('\nPotentially conflicting common tool names found:')
for name in tool_counts.keys():
    if any(common in name.lower() for common in common_names):
        print(f'  - {name}')