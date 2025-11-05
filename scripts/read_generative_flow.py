import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import json

m = importlib.import_module('server')

# Grant access first
print("=== Granting access to E:\\Projects\\generative_flow ===")
r_grant = m.handle_message({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'grant_directory_access','arguments':{'directory':'E:\\Projects\\generative_flow'}}})
print('Grant result:', json.loads(r_grant.get('result', {}).get('content', [{}])[0].get('text', '{}')))

# Read key files
files_to_read = [
    'E:\\Projects\\generative_flow\\README.md',
    'E:\\Projects\\generative_flow\\main.py',
    'E:\\Projects\\generative_flow\\requirements.txt',
    'E:\\Projects\\generative_flow\\config.py'
]

for file_path in files_to_read:
    print(f"\n=== Reading {file_path} ===")
    r = m.handle_message({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'read_file_content','arguments':{'file_path':file_path}}})
    
    if 'error' in r:
        print(f'ERROR: {r["error"]}')
    else:
        content = r.get('result', {}).get('content', [{}])[0].get('text', '')
        if content:
            lines = content.split('\n')[:30]  # First 30 lines
            for i, line in enumerate(lines, 1):
                print(f'{i:3}: {line}')
            if len(content.split('\n')) > 30:
                print('... (truncated)')
        else:
            print('(empty file)')
