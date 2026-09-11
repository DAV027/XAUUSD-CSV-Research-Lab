from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {"completed_ids": []}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"completed_ids": []}
        if not isinstance(value, dict):
            return {"completed_ids": []}
        completed = value.get("completed_ids", [])
        if not isinstance(completed, list):
            completed = []
        return {**value, "completed_ids": [str(item) for item in completed]}

    def write(self, completed_ids: Iterable[str], **metadata) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed_ids": [str(item) for item in completed_ids],
            **metadata,
        }
        temp_path = self.path.with_name(self.path.name + ".tmp")
        data = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)


__all__ = ["CheckpointStore"]
