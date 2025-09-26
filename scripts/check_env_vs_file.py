import os, sys
from pathlib import Path

ENV_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.secrets/.env.local')

KEYS = [
  'ANTHROPIC_API_KEY','ANTHROPIC_MODEL','ANTHROPIC_BASE_URL','ANTHROPIC_VERSION',
  'OPENAI_API_KEY','OPENAI_MODEL','OPENAI_BASE_URL',
  'LMSTUDIO_API_KEY','LMSTUDIO_API_BASE','LM_STUDIO_URL','LMSTUDIO_MODEL','MODEL_NAME'
]

def parse_line(s: str):
    s = s.strip()
    if not s or s.startswith('#'): return None
    if s.startswith('export '): s = s[len('export '):].strip()
    if '=' not in s: return None
    k, v = s.split('=', 1)
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1]
    return k.strip(), v

def mask(v: str):
    if not v: return '<empty>'
    return v[:4] + '...' + v[-4:]

pairs = {}
if not ENV_FILE.exists():
    print(f"Env file not found: {ENV_FILE}")
    sys.exit(1)
for raw in ENV_FILE.read_text(encoding='utf-8').splitlines():
    kv = parse_line(raw)
    if kv: pairs[kv[0]] = kv[1]

print(f"Comparing environment vs {ENV_FILE}")
all_ok = True
for k in KEYS:
    file_v = pairs.get(k, '')
    env_v = os.getenv(k, '')
    eq = (file_v == env_v)
    all_ok = all_ok and (eq or not file_v)  # allow file missing key
    print(f"{k:20} file={mask(file_v)} env={mask(env_v)} match={'YES' if eq else 'NO'}")

sys.exit(0 if all_ok else 2)

