# app/core/storage.py
import logging
import pickle
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from app import config
from app.logging_config import get_logger, timed

log = get_logger(_name_)


# ────────────────────────────────────────────────────────────────
# Dataclass for a single embedding record
# ────────────────────────────────────────────────────────────────
@dataclass
class EmbRecord:
    person_id: str
    embedding: np.ndarray
    ts: float
    source: str


# ────────────────────────────────────────────────────────────────
# Local embedding storage manager
# ────────────────────────────────────────────────────────────────
class LocalStore:
    """
    Handles local storage of embeddings and aligned face crops.
    Embeddings are stored as pickle (index.pkl), crops as JPEGs.
    """

    def _init_(self):
        config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.index_file = config.EMBED_INDEX_FILE
        self._records: list[EmbRecord] = []
        self._matrix: np.ndarray | None = None
        self._person: list[str] = []
        self._load()

    # ─────────────────────────────────────────────────────────────
    def _load(self):
        """Load existing embeddings from disk into memory."""
        if self.index_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self._records = [
                EmbRecord(
                    r["person_id"],
                    np.asarray(r["embedding"], np.float32),
                    r["ts"],
                    r["source"],
                )
                for r in data
            ]
            log.info("embeddings_loaded", extra={"count": len(self._records)})
        else:
            log.info("no_existing_index", extra={"file": str(self.index_file)})
        self._rebuild_cache()

    # ─────────────────────────────────────────────────────────────
    def _rebuild_cache(self):
        """Rebuild in-memory matrix and person list for cosine search."""
        if self._records:
            self._matrix = np.vstack([r.embedding for r in self._records])
            self._person = [r.person_id for r in self._records]
        else:
            self._matrix = np.zeros((0, config.EMB_DIM), np.float32)
            self._person = []
        log.info(
            "cache_rebuilt",
            extra={"records": len(self._records), "unique_people": len(set(self._person))},
        )

    # ─────────────────────────────────────────────────────────────
    def persist(self):
        """Persist all records to pickle."""
        with timed(log, "persist_index", count=len(self._records), file=str(self.index_file)):
            serializable = [
                {
                    "person_id": r.person_id,
                    "embedding": r.embedding.astype(np.float32),
                    "ts": r.ts,
                    "source": r.source,
                }
                for r in self._records
            ]
            with open(self.index_file, "wb") as f:
                pickle.dump(serializable, f)
        log.info("persist_done", extra={"count": len(self._records)})

    # ─────────────────────────────────────────────────────────────
    def save_aligned(self, person_id: str, aligned_bgr: np.ndarray) -> Path:
        """
        Previously saved aligned face crops to disk.
        Now a no-op because faces are stored in Firebase via /register.
        """
        log.info("save_crop_skipped", extra={"person_id": person_id})
        # Return a dummy path so existing callers don't break
        return config.DATA_DIR / "faces" / f"{person_id}_ignored.jpg"



    # ─────────────────────────────────────────────────────────────
    def add_embeddings(self, person_id: str, embs: np.ndarray, source: str) -> int:
        """Add new embeddings and update cache."""
        if embs is None or embs.size == 0:
            log.warning("add_embs_empty", extra={"person_id": person_id})
            return 0

        now = time.time()
        for e in embs:
            self._records.append(EmbRecord(person_id, e, now, source))
        log.info("add_embs", extra={"person_id": person_id, "count": int(embs.shape[0])})
        self.persist()
        self._rebuild_cache()
        return embs.shape[0]

    # ─────────────────────────────────────────────────────────────
    def search_cosine(self, q: np.ndarray, top_k: int):
        """
        Return, for each query embedding, a ranked list of unique persons:
        [{person_id, score}, ...]
        """
        if self._matrix is None or len(self._records) == 0:
            log.info("search_empty")
            return []

        with timed(log, "search_cosine", queries=int(q.shape[0]), db_size=len(self._records)):
            sims = q @ self._matrix.T  # cosine sim since embeddings are L2-normalized

        results = []
        for i in range(q.shape[0]):
            agg = {}
            for j, pid in enumerate(self._person):
                s = float(sims[i, j])
                if (pid not in agg) or (s > agg[pid]):
                    agg[pid] = s
            ranked = sorted(
                ({"person_id": pid, "score": sc} for pid, sc in agg.items()),
                key=lambda x: -x["score"],
            )[:top_k]
            results.append(ranked)

        log.info(
            "search_done",
            extra={
                "queries": int(q.shape[0]),
                "unique_people": len(set(self._person)),
                "top_k": top_k,
            },
        )
        return results

    # ─────────────────────────────────────────────────────────────
    def people(self) -> dict[str, int]:
        """Return dict of person_id → embedding count."""
        counts = {}
        for r in self._records:
            counts[r.person_id] = counts.get(r.person_id, 0) + 1
        log.info("people_summary", extra={"unique_people": len(counts)})
        return counts

    # ─────────────────────────────────────────────────────────────
    def delete_person(self, person_id: str) -> int:
        """Delete all embeddings and crops for a person."""
        before = len(self._records)
        self._records = [r for r in self._records if r.person_id != person_id]
        pid_dir = config.FACES_DIR / person_id
        if pid_dir.exists():
            for p in pid_dir.glob("*.jpg"):
                p.unlink(missing_ok=True)
            pid_dir.rmdir()
        self.persist()
        self._rebuild_cache()
        removed = before - len(self._records)
        log.info("delete_person", extra={"person_id": person_id, "removed": removed})
        return removed