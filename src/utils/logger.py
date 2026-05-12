from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_dir(root: str | Path, prefix: str) -> Path:
    root = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        row = {k: _to_scalar(v) for k, v in row.items()}

        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        for key in row:
            if key not in self.fieldnames:
                self.fieldnames.append(key)
                self._rewrite_with_new_header()

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def _rewrite_with_new_header(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _to_scalar(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except Exception:
        pass

    return value
