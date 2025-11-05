import server


def test_tools_list_returns_tools():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    res = server.handle_message(msg)
    assert isinstance(res, dict)
    assert res.get("jsonrpc") == "2.0"
    assert res.get("id") == 1
    result = res.get("result")
    assert isinstance(result, dict)
    tools = result.get("tools")
    assert isinstance(tools, list) and len(tools) > 5
    # ensure tool schema key is present in either snake_case (MCP) or camelCase (internal)
    assert any(("input_schema" in t or "inputSchema" in t) for t in tools)

