"""Pamięć podręczna na dysku.

Każde źródło zapisuje wynik pod kluczem (źródło, nazwa). Dzięki temu raport i testy
można powtórzyć na dokładnie tych samych danych, bez dostępu do sieci.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


class CacheMiss(LookupError):
    """Brak danych w pamięci podręcznej w trybie offline."""


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class DiskCache:
    def __init__(self, root: Path, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline

    def _path(self, source: str, key: str, suffix: str) -> Path:
        return self.root / _safe(source) / f"{_safe(key)}{suffix}"

    def frame(self, source: str, key: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self._path(source, key, ".parquet")
        if path.exists():
            return pd.read_parquet(path)
        if self.offline:
            raise CacheMiss(f"{source}/{key}")
        df = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    def json(self, source: str, key: str, fetch: Callable[[], Any]) -> Any:
        path = self._path(source, key, ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise CacheMiss(f"{source}/{key}")
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
