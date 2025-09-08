import json, os, sys
# Ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from server import get_server_singleton, handle_tool_call

server = get_server_singleton()

# 1) router_config
msg1 = {"params": {"name": "router_config", "arguments": {}}}
res1 = handle_tool_call(msg1)
print("ROUTER_CONFIG RESULT:\n" + json.dumps(res1, indent=2) + "\n")

# 2) health_check with probe_providers=true
msg2 = {"params": {"name": "health_check", "arguments": {"probe_providers": True}}}
res2 = handle_tool_call(msg2)
print("HEALTH_CHECK RESULT:\n" + json.dumps(res2, indent=2) + "\n")

# 3) chat_with_tools diagnostic test: instruct the model to output a single space so .strip() -> ""
msg3 = {"params": {"name": "chat_with_tools", "arguments": {
  "instruction": "Respond with a single space character and no other text.",
  "allowed_tools": [],
  "max_iters": 1,
  "temperature": 0.0,
  "max_tokens": 16
}}}
res3 = handle_tool_call(msg3)
print("CHAT_WITH_TOOLS DIAG TEST:\n" + json.dumps(res3, indent=2) + "\n")

