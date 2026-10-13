"""Pamięć podręczna na dysku.

Każde źródło zapisuje wynik pod kluczem (źródło, nazwa). Dzięki temu raport i testy
można powtórzyć na dokładnie tych samych danych, bez dostępu do sieci.

Dane, które się zmieniają (notowania, makro, sprawozdania), mają maksymalny wiek (max_age):
starszy plik jest pobierany ponownie. Jeśli pobranie się nie uda, używany jest zapisany plik,
a data danych trafia do raportu. W trybie offline wiek nie ma znaczenia.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import timedelta
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

    def _fresh(self, path: Path, max_age: timedelta | None) -> bool:
        if not path.exists():
            return False
        if self.offline or max_age is None:
            return True
        return time.time() - path.stat().st_mtime <= max_age.total_seconds()

    def _load(self, path: Path, read, write, fetch, source: str, key: str, max_age):
        if self._fresh(path, max_age):
            return read(path)
        if self.offline:
            raise CacheMiss(f"{source}/{key}")
        try:
            data = fetch()
        except Exception:
            # Brak sieci lub błąd źródła: lepiej starsze dane (z datą w raporcie) niż żadne.
            if path.exists():
                return read(path)
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, data)
        return data

    def frame(
        self,
        source: str,
        key: str,
        fetch: Callable[[], pd.DataFrame],
        max_age: timedelta | None = None,
    ) -> pd.DataFrame:
        path = self._path(source, key, ".parquet")
        return self._load(
            path, pd.read_parquet, lambda p, df: df.to_parquet(p), fetch, source, key, max_age
        )

    def json(
        self, source: str, key: str, fetch: Callable[[], Any], max_age: timedelta | None = None
    ) -> Any:
        path = self._path(source, key, ".json")
        return self._load(
            path,
            lambda p: json.loads(p.read_text(encoding="utf-8")),
            lambda p, data: p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8"),
            fetch,
            source,
            key,
            max_age,
        )
