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


# 4) chat_with_tools with tool_choice='required' and no tools to verify resilience
msg4 = {"params": {"name": "chat_with_tools", "arguments": {
  "instruction": "Summarize: hello world.",
  "allowed_tools": [],
  "tool_choice": "required",
  "max_iters": 1,
  "temperature": 0.1,
  "max_tokens": 64
}}}
res4 = handle_tool_call(msg4)
print("CHAT_WITH_TOOLS REQUIRED+NO-TOOLS TEST:\n" + json.dumps(res4, indent=2) + "\n")

# 5) smart_task improves ambiguous selection for API-like instructions
msg5 = {"params": {"name": "smart_task", "arguments": {
  "instruction": "curl: curl -sS -X POST \"$GW/chemberta/predict\" -H \"Content-Type: application/json\" -d '{\"smiles\":\"CCO\"}'\nPowerShell: Invoke-RestMethod -Method POST -Uri \"$GW/chemberta/predict\" -ContentType 'application/json' -Body '{\"smiles\":\"CCO\"}'",
  "dry_run": True
}}}
res5 = handle_tool_call(msg5)
print("SMART_TASK API-LIKE INSTRUCTION TEST:\n" + json.dumps(res5, indent=2) + "\n")
