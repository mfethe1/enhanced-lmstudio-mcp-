import json
import os
import server

if __name__ == "__main__":
    args = {"probe_lm": True, "probe_providers": True}
    res = server.handle_health_check(args, server.EnhancedLMStudioMCPServer())
    print(json.dumps(res, indent=2))

