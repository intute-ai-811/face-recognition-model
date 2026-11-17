# app/core/sqlite_store.py
"""
SQLite enrollment store (schema v1).

Same public interface and the same matching code as the pickle LocalStore, plus:
  - atomic transactions (WAL, synchronous=FULL) instead of whole-file rewrites;
  - every embedding is tagged with the model that produced it; only rows of the
    active model are ever loaded, so embeddings from different models can never mix;
  - deleting a person cascades to their embeddings;
  - an audit_log of enroll / rename / delete actions (no biometric data in it).

Optional pickle mirror: while STORAGE_MIRROR_PICKLE=1 (default), every write is
also applied to the old pickle files, so switching STORAGE_BACKEND back to
"pickle" loses nothing. Turn it off once SQLite has been stable for a while.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from app import config
from app.core.storage import build_snapshot, search_snapshot, snapshot_counts
from app.logging_config import get_logger, timed

log = get_logger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS people (
    person_id  TEXT PRIMARY KEY,
    name       TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    consent_at REAL
);
CREATE TABLE IF NOT EXISTS models (
    model_id   TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL UNIQUE,
    emb_dim    INTEGER NOT NULL,
    threshold  REAL,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY,
    person_id  TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL REFERENCES models(model_id),
    vector     BLOB NOT NULL,
    source     TEXT,
    quality    REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_embeddings_model_person ON embeddings(model_id, person_id);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,
    actor     TEXT,
    action    TEXT NOT NULL,
    person_id TEXT,
    detail    TEXT
);
"""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_id_for(model_path: Path, sha256: str) -> str:
    return f"{Path(model_path).stem}:{sha256[:16]}"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] != SCHEMA_VERSION:
        raise RuntimeError(f"database schema v{row[0]} but this code expects v{SCHEMA_VERSION}")


class SqliteStore:
    def __init__(
        self,
        db_path: Path,
        model_id: str,
        model_sha256: str,
        emb_dim: int | None = None,
        mirror=None,
    ):
        self.db_path = Path(db_path)
        self.emb_dim = int(emb_dim or config.EMB_DIM)
        self.model_id = model_id
        self.model_sha256 = model_sha256
        self.mirror = mirror
        self._lock = threading.RLock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = connect(self.db_path)
        with self._lock:
            init_schema(self._conn)
            self._ensure_model()
            self._load()

    # ── setup ────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        row = self._conn.execute(
            "SELECT model_id, emb_dim FROM models WHERE sha256 = ?", (self.model_sha256,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO models(model_id, sha256, emb_dim, threshold, status, created_at) VALUES (?,?,?,?,?,?)",
                (self.model_id, self.model_sha256, self.emb_dim, config.SIMILARITY_THRESHOLD, "active", time.time()),
            )
        elif row[0] != self.model_id or row[1] != self.emb_dim:
            raise RuntimeError(f"model row mismatch for sha {self.model_sha256[:16]}: {row}")

    def _load(self) -> None:
        with self._lock:
            people = self._conn.execute("SELECT person_id, name FROM people").fetchall()
            self._names = {pid: name for pid, name in people if name}
            rows = self._conn.execute(
                "SELECT person_id, vector FROM embeddings WHERE model_id = ? ORDER BY id", (self.model_id,)
            ).fetchall()
            other = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_id != ?", (self.model_id,)
            ).fetchone()[0]

            self._pids: list[str] = []
            self._vecs: list[np.ndarray] = []
            for pid, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape != (self.emb_dim,):
                    raise ValueError(f"embedding for {pid} has {vec.size} values, expected {self.emb_dim}")
                self._pids.append(pid)
                self._vecs.append(vec)
            self._rebuild()

        log.info(
            "sqlite_store_loaded",
            extra={
                "db": str(self.db_path), "model_id": self.model_id, "embeddings": len(rows),
                "people": len(set(self._pids)), "other_model_embeddings_ignored": other,
            },
        )

    def _rebuild(self) -> None:
        with self._lock:
            self._snapshot = build_snapshot(list(self._vecs), list(self._pids), self.emb_dim)

    def _audit(self, action: str, person_id: str | None, detail: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO audit_log(ts, actor, action, person_id, detail) VALUES (?,?,?,?,?)",
            (time.time(), "api", action, person_id, json.dumps(detail) if detail else None),
        )

    def _mirror(self, method: str, *args) -> None:
        if self.mirror is None:
            return
        try:
            getattr(self.mirror, method)(*args)
        except Exception as e:  # never fail a request because the rollback copy failed
            log.error("pickle_mirror_failed", extra={"method": method, "error": repr(e)})

    # ── public interface (same as LocalStore) ────────────────────────────────

    def set_person(self, person_id: str, name: str | None):
        pid = str(person_id).strip()
        nm = (name or "").strip()
        if not pid or not nm:
            return
        now = time.time()
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT INTO people(person_id, name, created_at, updated_at) VALUES (?,?,?,?) "
                    "ON CONFLICT(person_id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
                    (pid, nm, now, now),
                )
                self._audit("set_name", pid)
            self._names[pid] = nm
        self._mirror("set_person", pid, nm)

    def get_person_name(self, person_id: str) -> str | None:
        pid = str(person_id).strip()
        return self._names.get(pid) if pid else None

    def add_embeddings(self, person_id: str, embs: np.ndarray, source: str) -> int:
        if embs is None or embs.size == 0:
            log.warning("add_embs_empty", extra={"person_id": person_id})
            return 0
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.ndim != 2:
            raise ValueError(f"add_embeddings expected 2D array, got shape {embs.shape}")
        if embs.shape[1] != self.emb_dim:
            raise ValueError(f"add_embeddings received dim={embs.shape[1]} but store emb_dim={self.emb_dim}")

        pid = str(person_id)
        now = time.time()
        vecs = [np.ascontiguousarray(e, dtype=np.float32) for e in embs]
        with self._lock:
            with timed(log, "sqlite_add_embeddings", count=len(vecs)):
                with self._conn:
                    self._conn.execute("BEGIN IMMEDIATE")
                    self._conn.execute(
                        "INSERT OR IGNORE INTO people(person_id, name, created_at, updated_at) VALUES (?,?,?,?)",
                        (pid, None, now, now),
                    )
                    self._conn.executemany(
                        "INSERT INTO embeddings(person_id, model_id, vector, source, created_at) VALUES (?,?,?,?,?)",
                        [(pid, self.model_id, v.tobytes(), source, now) for v in vecs],
                    )
                    self._audit("enroll", pid, {"embeddings": len(vecs)})
            self._pids.extend([pid] * len(vecs))
            self._vecs.extend(vecs)
            self._rebuild()
        self._mirror("add_embeddings", pid, embs, source)
        log.info("add_embs", extra={"person_id": pid, "count": len(vecs), "emb_dim": self.emb_dim})
        return len(vecs)

    def search_cosine(self, q: np.ndarray, top_k: int) -> list[list[dict]]:
        return search_snapshot(self._snapshot, q, top_k, self.emb_dim)

    def stats(self) -> dict[str, int]:
        """Counts for metrics/readiness, without logging (called on every scrape)."""
        ids = self._snapshot.person_ids
        return {"embeddings": len(ids), "people": len(set(ids))}

    def people(self) -> dict[str, int]:
        return snapshot_counts(self._snapshot)

    def delete_person(self, person_id: str) -> int:
        pid = str(person_id).strip()
        if not pid:
            return 0
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                removed = self._conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE person_id = ? AND model_id = ?", (pid, self.model_id)
                ).fetchone()[0]
                self._conn.execute("DELETE FROM people WHERE person_id = ?", (pid,))  # cascades to embeddings
                self._audit("delete", pid, {"embeddings": removed})
            keep = [i for i, p in enumerate(self._pids) if p != pid]
            self._pids = [self._pids[i] for i in keep]
            self._vecs = [self._vecs[i] for i in keep]
            self._names.pop(pid, None)
            self._rebuild()
        self._mirror("delete_person", pid)
        log.info("delete_person", extra={"person_id": pid, "removed_embeddings": removed})
        return int(removed)

    def save_aligned(self, person_id: str, aligned_img) -> None:
        from app.core.storage import save_aligned_crop
        save_aligned_crop(person_id, aligned_img)

    def reload(self) -> None:
        log.info("store_reload")
        self._load()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
