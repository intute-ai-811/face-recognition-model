# app/core/store_factory.py
"""
Selects the enrollment storage backend.

  STORAGE_BACKEND=pickle  (default) data/embeddings/index.pkl + data/people.pkl
  STORAGE_BACKEND=sqlite  data/attendance.db, created by scripts/migrate_pickle_to_sqlite.py
  STORAGE_MIRROR_PICKLE=1 (default, sqlite only) also apply every write to the pickles,
                          so switching back to "pickle" loses nothing
"""
from __future__ import annotations

import os
from pathlib import Path

from app import config
from app.logging_config import get_logger

log = get_logger(__name__)


def create_store():
    backend = os.getenv("STORAGE_BACKEND", "pickle").strip().lower()
    if backend == "pickle":
        from app.core.storage import LocalStore
        return LocalStore()
    if backend == "sqlite":
        return create_sqlite_store()
    raise ValueError(f"STORAGE_BACKEND must be 'pickle' or 'sqlite', got {backend!r}")


def create_sqlite_store():
    from app.core.sqlite_store import SqliteStore, file_sha256, model_id_for

    db = Path(config.DB_FILE)
    if not db.exists() and Path(config.EMBED_INDEX_FILE).exists():
        # Refuse to start with an empty database next to existing enrollments:
        # everyone would silently become "unknown".
        raise RuntimeError(
            f"STORAGE_BACKEND=sqlite but {db} does not exist while {config.EMBED_INDEX_FILE} does. "
            "Run scripts/migrate_pickle_to_sqlite.py first."
        )

    model_path = Path(config.MOBILEFACENET_MODEL)
    sha = file_sha256(model_path)

    mirror = None
    if os.getenv("STORAGE_MIRROR_PICKLE", "1") == "1":
        try:
            from app.core.storage import LocalStore
            mirror = LocalStore()
        except Exception as e:
            log.error("pickle_mirror_unavailable", extra={"error": repr(e)})

    store = SqliteStore(db, model_id=model_id_for(model_path, sha), model_sha256=sha, mirror=mirror)
    log.info("storage_backend", extra={"backend": "sqlite", "db": str(db), "pickle_mirror": mirror is not None})
    return store
