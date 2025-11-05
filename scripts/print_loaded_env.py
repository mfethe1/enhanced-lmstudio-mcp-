import os
from server import _load_local_secrets  # ensures loader is available

# Re-load to be explicit in this process
_load_local_secrets()

keys = [
  'ANTHROPIC_API_KEY','ANTHROPIC_MODEL','ANTHROPIC_BASE_URL','ANTHROPIC_VERSION',
  'OPENAI_API_KEY','OPENAI_MODEL','OPENAI_BASE_URL',
  'LMSTUDIO_API_KEY','LMSTUDIO_API_BASE','LM_STUDIO_URL','LMSTUDIO_MODEL','MODEL_NAME'
]

def mask(v: str):
    if not v:
        return '<empty>'
    return v[:4] + '...' + v[-4:]

for k in keys:
    print(f"{k:20} {mask(os.getenv(k, ''))}")

