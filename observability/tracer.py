from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator


class ExecutionTracer:
    """Very small tracing helper that records durations.

    Replace with OpenTelemetry or similar later.
    """

    def __init__(self) -> None:
        self.events = []

    @contextlib.contextmanager
    def trace(self, name: str, args: Dict[str, Any]) -> Iterator[None]:
        start = time.time()
        try:
            yield
        finally:
            self.events.append({"name": name, "duration": time.time() - start})

