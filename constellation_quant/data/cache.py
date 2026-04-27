"""Hash-based cache for processed data / features / graphs.

Usage pattern:

    cache = CacheManager(paths.cache_dir)
    key = cache.compute_key(
        {"feature": feature_cfg, "split": "train"},
        source_files=[paths.price_file(t) for t in tickers],
    )
    cached = cache.get(key)
    if cached is None:
        out = expensive_build(...)
        cache.put(key, out)
    else:
        out = cached

Invalidation: the key incorporates (a) a hash of the serialised config and
(b) the mtimes of any source files passed in. If either changes, the key
changes and the miss triggers a rebuild.

Storage formats:
    - `.pt`  for torch tensors (via torch.save / torch.load)
    - `.pq`  for pandas DataFrames (parquet)
    - `.pkl` for arbitrary Python objects (fallback)

Format is chosen from the value type at `put()` time and recorded in a
sidecar metadata file so `get()` can restore correctly.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: Path
    fmt: str
    bytes: int


class CacheManager:
    """File-backed cache with deterministic key computation."""

    META_SUFFIX = ".meta.json"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    # ── Key computation ────────────────────────────────────────────────

    @staticmethod
    def compute_key(
        config: Mapping[str, Any],
        source_files: Optional[Iterable[Path]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Deterministic cache key.

        Keyed on (sorted JSON of config) + (sorted list of source file stats).
        Two identical configs over identical source files -> identical key.
        """
        h = hashlib.sha256()
        payload = {
            "config": _canonicalise(config),
            "extra":  _canonicalise(extra or {}),
            "sources": sorted(_file_stats(source_files or [])),
        }
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()[:20]

    # ── CRUD ───────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        data_path = self.cache_dir / meta["filename"]
        if not data_path.exists():
            return None
        return self._load(data_path, meta["format"])

    def put(self, key: str, value: Any) -> CacheEntry:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fmt = self._pick_format(value)
        ext = {"torch": ".pt", "parquet": ".pq", "pickle": ".pkl"}[fmt]
        data_path = self.cache_dir / f"{key}{ext}"
        self._save(value, data_path, fmt)

        meta = {"key": key, "format": fmt, "filename": data_path.name}
        self._meta_path(key).write_text(json.dumps(meta, indent=2))
        return CacheEntry(key=key, path=data_path, fmt=fmt, bytes=data_path.stat().st_size)

    def contains(self, key: str) -> bool:
        return self._meta_path(key).exists()

    def invalidate(self, key: str) -> bool:
        """Delete a cached entry. Returns True if anything was removed."""
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        data_path = self.cache_dir / meta["filename"]
        data_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return True

    def clear(self) -> int:
        """Delete every cache entry. Returns the number of entries removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for meta in self.cache_dir.glob(f"*{self.META_SUFFIX}"):
            key = meta.name[: -len(self.META_SUFFIX)]
            if self.invalidate(key):
                count += 1
        return count

    # ── Format handling ────────────────────────────────────────────────

    @staticmethod
    def _pick_format(value: Any) -> str:
        # Late imports — torch/pandas are heavy.
        try:
            import torch
            if isinstance(value, torch.Tensor) or (
                isinstance(value, dict)
                and value
                and all(isinstance(v, torch.Tensor) for v in value.values())
            ):
                return "torch"
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                return "parquet"
        except ImportError:
            pass
        return "pickle"

    @staticmethod
    def _save(value: Any, path: Path, fmt: str) -> None:
        if fmt == "torch":
            import torch
            torch.save(value, path)
        elif fmt == "parquet":
            value.to_parquet(path, index=False)
        else:
            with path.open("wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(path: Path, fmt: str) -> Any:
        if fmt == "torch":
            import torch
            return torch.load(path, weights_only=False)
        if fmt == "parquet":
            import pandas as pd
            return pd.read_parquet(path)
        with path.open("rb") as f:
            return pickle.load(f)

    # ── Internals ──────────────────────────────────────────────────────

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{self.META_SUFFIX}"


# ── Canonicalisation helpers ───────────────────────────────────────────────


def _canonicalise(obj: Any) -> Any:
    """Recursively sort dict keys and convert Paths/sets to JSON-safe types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(_canonicalise(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    return obj


def _file_stats(paths: Iterable[Path]) -> list[tuple[str, int, int]]:
    """Return (name, mtime_ns, size) triples for existing files. Missing -> (name, 0, 0)."""
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            st = p.stat()
            out.append((p.name, int(st.st_mtime_ns), int(st.st_size)))
        else:
            out.append((p.name, 0, 0))
    return out
