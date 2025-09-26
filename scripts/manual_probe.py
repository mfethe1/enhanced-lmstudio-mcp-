import os, json, sys, requests
from pathlib import Path

def parse_env_file(path: Path) -> dict:
    pairs = {}
    if not path.exists():
        return pairs
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("export "): s = s[len("export "):].strip()
        if "=" not in s: continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        pairs[k] = v
    return pairs

def choose(v_file: str, v_env: str, default: str = "") -> str:
    return v_file or v_env or default

def load_config():
    env = parse_env_file(Path('.secrets/.env.local'))
    # Build a simple accessor
    def g(k, default=""):
        return env.get(k) or os.getenv(k) or default
    return {
        # LM Studio
        "LMSTUDIO_API_BASE": g("LMSTUDIO_API_BASE", "http://localhost:1234/v1").rstrip("/"),
        "LMSTUDIO_API_KEY": g("LMSTUDIO_API_KEY", g("OPENAI_API_KEY", "sk-noauth")),
        "LMSTUDIO_MODEL": env.get("LMSTUDIO_MODEL") or os.getenv("LMSTUDIO_MODEL") or os.getenv("MODEL_NAME") or "openai/gpt-oss-20b",
        # OpenAI
        "OPENAI_API_KEY": g("OPENAI_API_KEY", ""),
        "OPENAI_BASE_URL": g("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "OPENAI_MODEL": env.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_FALLBACK_MODEL") or "gpt-4o-mini",
        # Anthropic
        "ANTHROPIC_API_KEY": g("ANTHROPIC_API_KEY", ""),
        "ANTHROPIC_BASE_URL": g("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/"),
        "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
        "ANTHROPIC_VERSION": g("ANTHROPIC_VERSION", "2023-06-01"),
    }

def probe_lmstudio(cfg):
    base = cfg["LMSTUDIO_API_BASE"]
    model = cfg["LMSTUDIO_MODEL"]
    chat_url = f"{base}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['LMSTUDIO_API_KEY']}"}
    try:
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

def probe_openai(cfg):
    key = cfg["OPENAI_API_KEY"]
    if not key:
        print("OPENAI SKIP: no OPENAI_API_KEY in env or file")
        return
    base = cfg["OPENAI_BASE_URL"]
    model = cfg["OPENAI_MODEL"]
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

def probe_anthropic(cfg):
    key = cfg["ANTHROPIC_API_KEY"]
    if not key:
        print("ANTHROPIC SKIP: no ANTHROPIC_API_KEY in env or file")
        return
    base = cfg["ANTHROPIC_BASE_URL"]
    model = cfg["ANTHROPIC_MODEL"]
    version = cfg["ANTHROPIC_VERSION"]
    url = f"{base}/messages"
    payload = {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    try:
        r = requests.post(url, json=payload, headers={"x-api-key": key, "anthropic-version": version, "content-type": "application/json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        parts = data.get("content", [])
        txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        print(f"ANTHROPIC OK model={model} reply={txt[:60]!r}")
    except Exception as e:
        print(f"ANTHROPIC ERROR: {e}")

if __name__ == "__main__":
    cfg = load_config()
    probe_lmstudio(cfg)
    probe_openai(cfg)
    probe_anthropic(cfg)

