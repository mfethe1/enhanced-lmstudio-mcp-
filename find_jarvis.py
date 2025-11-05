with open('server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find lines with 'jarvis' in tool name context
for i, line in enumerate(lines, 1):
    if '"name": "jarvis"' in line or '"name":"jarvis"' in line:
        # Show context
        start = max(0, i-3)
        end = min(len(lines), i+3)
        print(f'Found at line {i}:')
        for j in range(start, end):
            prefix = '>>> ' if j == i-1 else '    '
            print(f'{prefix}{j+1}: {lines[j].rstrip()}')
        print()