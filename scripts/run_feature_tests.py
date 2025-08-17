import re
import sys
import types
import json

import server


def phase1_multi_chat_tests():
    print("--- Phase 1: Multi-chat safety tests ---")
    sv = server.get_server_singleton()

    # 1) No session_id provided
    out = server.handle_sequential_thinking({
        "thought": "Start analysis",
        "thought_number": 1,
        "total_thoughts": 3,
        "next_thought_needed": False,
    }, sv)
    assert isinstance(out, dict), f"Unexpected output type: {type(out)}"
    sid = out.get("session_id")
    assert isinstance(sid, str) and sid.startswith("thinking_") and len(sid) == len("thinking_") + 12, sid
    print("PASS: UUID-based session_id generated:", sid)

    # 2) Concurrent sessions (simulated): ensure no cross contamination
    out2 = server.handle_sequential_thinking({
        "thought": "Other session start",
        "thought_number": 1,
        "total_thoughts": 2,
        "next_thought_needed": False,
        "session_id": "thinking_testA",
    }, sv)
    out3 = server.handle_sequential_thinking({
        "thought": "Third session start",
        "thought_number": 1,
        "total_thoughts": 2,
        "next_thought_needed": False,
        "session_id": "thinking_testB",
    }, sv)
    histA = sv.storage.get_thinking_session("thinking_testA")
    histB = sv.storage.get_thinking_session("thinking_testB")
    assert len(histA) == 1 and "Other session start" in histA[0]["thought"], histA
    assert len(histB) == 1 and "Third session start" in histB[0]["thought"], histB
    print("PASS: No cross-contamination between sessions A and B")

    # 3) auto_next derives next index from storage history
    async def fake_llm(prompt, temperature=0.2, retries=2, backoff=0.5):
        return "AUTO_NEXT_SUGGESTION"

    sv.make_llm_request_with_retry = fake_llm  # monkey-patch
    sid2 = "thinking_autonext"
    server.handle_sequential_thinking({
        "thought": "First",
        "thought_number": 1,
        "total_thoughts": 3,
        "next_thought_needed": True,
        "auto_next": True,
        "session_id": sid2,
    }, sv)
    hist = sv.storage.get_thinking_session(sid2)
    nums = [h["thought_number"] for h in hist]
    assert 1 in nums and 2 in nums, hist
    print("PASS: auto_next stored next step with storage-derived index:", nums)


def phase1_deep_research_tests():
    print("\n--- Phase 1: Deep research tool tests ---")
    sv = server.get_server_singleton()

    # Stub Firecrawl MCP function to simulate Stage 1 success
    mod = types.ModuleType("functions")

    def fc_deep(args):
        return {"data": {"finalAnalysis": "Firecrawl: summary ok"}, "sources": ["https://example.com"]}

    mod.firecrawl_deep_research_firecrawl_mcp = fc_deep
    sys.modules["functions"] = mod

    res = server.handle_deep_research({"query": "Test topic", "max_depth": 3, "time_limit": 60}, sv)
    assert isinstance(res, str) and "Deep research" in res, res
    print("deep_research summary:", res[:200].replace("\n", " "))

    # Extract research_id and verify storage
    m = re.search(r"Deep research ([0-9a-f]{12}) completed", res)
    assert m, "research_id not found in summary"
    rid = m.group(1)
    rows = sv.storage.retrieve_memory(key=f"research_{rid}")
    assert rows, "research artifacts not stored"
    print("PASS: research artifacts stored under key research_%s" % rid)

    # get_research_details retrieves and pretty-prints
    details = server.handle_get_research_details({"research_id": rid}, sv)
    assert isinstance(details, str) and ("Firecrawl" in details or "finalAnalysis" in details), details
    print("PASS: get_research_details returns formatted artifacts for id", rid)

    # Missing research_id should raise ValidationError
    try:
        server.handle_get_research_details({"research_id": ""}, sv)
        raise AssertionError("Expected ValidationError for missing research_id")
    except server.ValidationError:
        print("PASS: missing research_id raises ValidationError")

    # Stage 2 fallback: ensure summary contains Stage 2 note even if CrewAI not installed
    if "crewai" in sys.modules:
        del sys.modules["crewai"]
    res2 = server.handle_deep_research({"query": "Another topic"}, sv)
    assert "Stage 2 (CrewAI):" in res2, res2
    print("PASS: CrewAI fallback path executed (or succeeded), summary includes Stage 2 note")


def phase1_mcp_compliance_tests():
    print("\n--- Phase 1: MCP compliance tests ---")

    # Verify envelope is text-only for deep_research
    msg = {"id": 99, "params": {"name": "deep_research", "arguments": {"query": "Compliance test"}}}
    wrapped = server.handle_tool_call(msg)
    assert "result" in wrapped and "content" in wrapped["result"], wrapped
    ct = wrapped["result"]["content"][0]
    assert ct["type"] == "text" and isinstance(ct["text"], str)
    print("PASS: MCP envelope contains text-only content for deep_research")

    # Verify envelope is text-only for get_research_details (use an unlikely ID)
    msg2 = {"id": 100, "params": {"name": "get_research_details", "arguments": {"research_id": "invalid"}}}
    wrapped2 = server.handle_tool_call(msg2)
    ct2 = wrapped2["result"]["content"][0]
    assert ct2["type"] == "text"
    print("PASS: MCP envelope contains text-only content for get_research_details")

    # Backward compatibility: call health_check
    msg3 = {"id": 101, "params": {"name": "health_check", "arguments": {}}}
    wrapped3 = server.handle_tool_call(msg3)
    ct3 = wrapped3["result"]["content"][0]
    assert ct3["type"] == "text"
    print("PASS: health_check works under new routing and returns text")


if __name__ == "__main__":
    phase1_multi_chat_tests()
    phase1_deep_research_tests()
    phase1_mcp_compliance_tests()
    print("\nALL TESTS PASSED")

