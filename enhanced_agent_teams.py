from __future__ import annotations

import os
from typing import Any

from enhanced_agent_architecture import decide_backend


def decide_backend_for_role(role: str, task_desc: str) -> str:
    return decide_backend(role, task_desc).backend


__all__ = ["decide_backend_for_role"]

