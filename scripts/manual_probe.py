import os, json, sys, requests

def probe_lmstudio():
    base = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1").rstrip("/")
    model = os.getenv("LMSTUDIO_MODEL", os.getenv("MODEL_NAME", "openai/gpt-oss-20b"))
    chat_url = f"{base}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    try:
        # Allow long first-token latency (model load); try up to 90s
        lkey = os.getenv("LMSTUDIO_API_KEY", os.getenv("OPENAI_API_KEY", "sk-noauth"))
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {lkey}"}
        r = requests.post(chat_url, json=payload, headers=headers, timeout=90)
        r.raise_for_status()
        data = r.json()
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"LMSTUDIO OK model={model} reply={txt[:60]!r}")
        return
    except Exception:
        pass
    # Retry with max_completion_tokens in case server rejects max_tokens
    try:
        payload2 = dict(payload)
        payload2.pop("max_tokens", None)
        payload2["max_completion_tokens"] = 8
        r2 = requests.post(chat_url, json=payload2, headers=headers, timeout=90)
        r2.raise_for_status()
        data = r2.json()
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"LMSTUDIO OK (fallback tokens) model={model} reply={txt[:60]!r}")
        return
    except Exception:
        pass
    # Try completions (instruct) endpoint with same model id
    try:
        compl_url = f"{base}/completions"
        c_payload = {"model": model, "prompt": "hi", "max_tokens": 8}
        r3 = requests.post(compl_url, json=c_payload, headers={"Content-Type": "application/json"}, timeout=90)
        r3.raise_for_status()
        data = r3.json()
        txt = data.get("choices", [{}])[0].get("text", "")
        print(f"LMSTUDIO OK (completions) model={model} reply={txt[:60]!r}")
        return
    except Exception:
        pass
    # Try alternate model if available
    try:
        models = requests.get(f"{base}/models", timeout=10).json().get("data", [])
        alt = None
        for m in models:
            mid = m.get("id")
            if mid and mid != model:
                alt = mid
                break
        if alt:
            r4 = requests.post(chat_url, json={"model": alt, "messages": [{"role": "user", "content": "hi"}]}, headers=headers, timeout=90)
            r4.raise_for_status()
            data = r4.json()
            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"LMSTUDIO OK (alt model) model={alt} reply={txt[:60]!r}")
            return
    except Exception as ealt:
        print(f"LMSTUDIO ERROR: {ealt}")

def probe_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI SKIP: no OPENAI_API_KEY in env")
        return
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"))
    url = f"{base}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    try:
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"OPENAI OK model={model} reply={txt[:60]!r}")
    except Exception as e:
        print(f"OPENAI ERROR: {e}")

def probe_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC SKIP: no ANTHROPIC_API_KEY in env")
        return
    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet")
    version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    url = f"{base}/messages"
    payload = {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    try:
        r = requests.post(url, json=payload, headers={"x-api-key": key, "anthropic-version": version, "content-type": "application/json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        # messages API returns content as a list of parts
        parts = data.get("content", [])
        txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        print(f"ANTHROPIC OK model={model} reply={txt[:60]!r}")
    except Exception as e:
        print(f"ANTHROPIC ERROR: {e}")

if __name__ == "__main__":
    probe_lmstudio()
    probe_openai()
    probe_anthropic()

