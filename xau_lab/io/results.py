from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Mapping

from xau_lab.io.checkpoint import CheckpointStore

ERROR_COLUMNS = ["experiment_id", "exception_type", "message", "traceback_file"]


class ResultStore:
    def __init__(
        self,
        result_root: str | Path,
        *,
        catalog_ids: Iterable[str] | None = None,
        checkpoint: CheckpointStore | None = None,
    ):
        self.result_root = Path(result_root)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.master_path = self.result_root / "MASTER_RESULTS.csv"
        self.errors_path = self.result_root / "ERRORS.csv"
        self.error_dir = self.result_root / "errors"
        self.catalog_ids = tuple(str(item) for item in (catalog_ids or ()))
        self.checkpoint = checkpoint
        self._lock = mp.Lock()
        self._completed_set = self._read_completed_ids()

    def _read_completed_ids(self) -> set[str]:
        if not self.master_path.exists() or self.master_path.stat().st_size == 0:
            return set()
        completed: set[str] = set()
        with self.master_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "experiment_id" not in reader.fieldnames:
                raise ValueError("MASTER_RESULTS.csv is missing experiment_id")
            for row in reader:
                experiment_id = (row.get("experiment_id") or "").strip()
                if experiment_id:
                    if experiment_id in completed:
                        raise ValueError(f"duplicate experiment_id already stored: {experiment_id}")
                    completed.add(experiment_id)
        return completed

    @property
    def completed_ids(self) -> list[str]:
        if not self.catalog_ids:
            return sorted(self._completed_set)
        ordered = [item for item in self.catalog_ids if item in self._completed_set]
        extras = sorted(self._completed_set.difference(self.catalog_ids))
        return ordered + extras

    def pending_ids(self) -> list[str]:
        return [item for item in self.catalog_ids if item not in self._completed_set]

    def _existing_header(self, path: Path) -> list[str] | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, None)

    def _append_csv_row(self, path: Path, row: Mapping[str, object], fixed_columns: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self._existing_header(path)
        fieldnames = list(fixed_columns or header or row.keys())
        extras = [key for key in row if key not in fieldnames]
        if extras:
            raise ValueError(f"row contains fields not present in CSV schema: {extras}")
        needs_header = header is None
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            if needs_header:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())

    def append_result(self, row: Mapping[str, object]) -> None:
        experiment_id = str(row.get("experiment_id") or "").strip()
        if not experiment_id:
            raise ValueError("result row requires experiment_id")
        with self._lock:
            if experiment_id in self._completed_set:
                raise ValueError(f"duplicate experiment_id: {experiment_id}")
            self._append_csv_row(self.master_path, row)
            self._completed_set.add(experiment_id)
            if self.checkpoint is not None:
                self.checkpoint.write(self.completed_ids)

    def append_error(
        self,
        *,
        experiment_id: str,
        exception_type: str,
        message: str,
        traceback_text: str,
    ) -> None:
        experiment_id = str(experiment_id).strip()
        if not experiment_id:
            raise ValueError("experiment_id is required")
        self.error_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.error_dir / f"{experiment_id}.txt"
        temp_trace = trace_path.with_name(trace_path.name + ".tmp")
        with temp_trace.open("w", encoding="utf-8", newline="") as handle:
            handle.write(traceback_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_trace, trace_path)

        row = {
            "experiment_id": experiment_id,
            "exception_type": str(exception_type),
            "message": str(message),
            "traceback_file": str(Path("errors") / trace_path.name),
        }
        with self._lock:
            self._append_csv_row(self.errors_path, row, fixed_columns=ERROR_COLUMNS)


__all__ = ["ERROR_COLUMNS", "ResultStore"]
