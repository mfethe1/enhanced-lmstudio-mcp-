import json
import sys
sys.path.append('.')
from server import get_server_singleton, handle_tool_call

if __name__ == '__main__':
    server = get_server_singleton()
    msg = {"params": {"name": "health_check", "arguments": {"probe_providers": True}}}
    res = handle_tool_call(msg)
    print(json.dumps(res, indent=2))

    msg2 = {"params": {"name": "router_config", "arguments": {}}}
    res2 = handle_tool_call(msg2)
    print("\n\nROUTER_CONFIG\n" + json.dumps(res2, indent=2))

