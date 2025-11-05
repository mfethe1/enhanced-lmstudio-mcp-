import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import json

m = importlib.import_module('server')

# Test list_allowed_directories
print("=== Testing list_allowed_directories ===")
r1 = m.handle_message({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'list_allowed_directories','arguments':{}}})
print('list_allowed_directories result:', r1.get('result'))

# Test grant_directory_access
print("\n=== Testing grant_directory_access ===")
r2 = m.handle_message({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'grant_directory_access','arguments':{'directory':'E:\\Projects\\generative_flow'}}})
print('grant_directory_access result:', r2.get('result'))

# Test list again to confirm
print("\n=== Testing list_allowed_directories after grant ===")
r3 = m.handle_message({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'list_allowed_directories','arguments':{}}})
print('list_allowed_directories result:', r3.get('result'))

# Test list_directory on the external path
print("\n=== Testing list_directory on E:\\Projects\\generative_flow ===")
r4 = m.handle_message({'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'list_directory','arguments':{'directory_path':'E:\\Projects\\generative_flow'}}})
if 'error' in r4:
    print('ERROR:', r4['error'])
else:
    print('SUCCESS - Directory listing available')
    result = r4.get('result', {})
    if isinstance(result, dict) and 'content' in result:
        content = result['content'][0]['text'] if result['content'] else ''
        lines = content.split('\n')[:10]  # First 10 lines
        for line in lines:
            print('  ', line)
    else:
        print('  Result:', result)
