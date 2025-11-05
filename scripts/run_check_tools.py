import importlib, json

m = importlib.import_module('server')
resp = m.handle_message({'jsonrpc':'2.0','id':1,'method':'tools/list'})
print('OK tools/list')
tools = [t.get('name') for t in resp.get('result',{}).get('tools',[])]
print('TOTAL_TOOLS', len(tools))
print('HAS grant_directory_access?', 'grant_directory_access' in tools)
print('HAS list_allowed_directories?', 'list_allowed_directories' in tools)

r2 = m.handle_message({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'list_allowed_directories','arguments':{}}})
print('list_allowed_directories ->', r2.get('result'))

