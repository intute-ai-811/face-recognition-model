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

log = get_logger(__name__)


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

    This version enforces a single embedding dimension (config.EMB_DIM)
    across all stored embeddings and all queries, to avoid dimension
    mismatch errors (e.g. 128 vs 512) during cosine search.
    """

    def __init__(self):
        if not hasattr(config, "EMB_DIM"):
            raise ValueError(
                "config.EMB_DIM is not set. Please define it to match your "
                "embedder output dimension."
            )

        self.emb_dim: int = int(config.EMB_DIM)

        config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.index_file = config.EMBED_INDEX_FILE
        self._records: list[EmbRecord] = []
        self._matrix: np.ndarray | None = None
        self._person: list[str] = []
        self._load()

    # ─────────────────────────────────────────────────────────────
    def _load(self):
        """Load existing embeddings from disk into memory and validate dimensions."""
        if self.index_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)

            self._records = []
            bad_dims = set()

            for r in data:
                emb = np.asarray(r["embedding"], np.float32)
                # We expect a flat vector of length emb_dim
                if emb.ndim != 1:
                    bad_dims.add(tuple(emb.shape))
                elif emb.shape[0] != self.emb_dim:
                    bad_dims.add(emb.shape[0])
                else:
                    self._records.append(
                        EmbRecord(
                            r["person_id"],
                            emb,
                            r["ts"],
                            r["source"],
                        )
                    )

            if bad_dims:
                # Fail fast with a clear error instead of matmul mismatch later
                raise ValueError(
                    f"Loaded embeddings from {self.index_file} with dimensions {bad_dims}, "
                    f"but expected emb_dim={self.emb_dim}. "
                    f"This usually happens after changing the embedder model "
                    f"without recreating the index. "
                    f"Delete the old index file ({self.index_file}) and re-enroll users."
                )

            log.info(
                "embeddings_loaded",
                extra={
                    "count": len(self._records),
                    "file": str(self.index_file),
                    "emb_dim": self.emb_dim,
                },
            )
        else:
            log.info("no_existing_index", extra={"file": str(self.index_file)})

        self._rebuild_cache()

    # ─────────────────────────────────────────────────────────────
    def _rebuild_cache(self):
        """Rebuild in-memory matrix and person list for cosine search."""
        if self._records:
            # Extra safety: validate dimensions before stacking
            for r in self._records:
                if r.embedding.ndim != 1 or r.embedding.shape[0] != self.emb_dim:
                    raise ValueError(
                        f"Record for person_id={r.person_id} has embedding shape "
                        f"{r.embedding.shape}, expected ({self.emb_dim},). "
                        f"The index file is inconsistent. Delete {self.index_file} "
                        f"and re-enroll."
                    )

            self._matrix = np.vstack([r.embedding for r in self._records])
            self._person = [r.person_id for r in self._records]
        else:
            # When empty, keep a consistent zero-sized matrix with emb_dim columns
            self._matrix = np.zeros((0, self.emb_dim), np.float32)
            self._person = []

        log.info(
            "cache_rebuilt",
            extra={
                "records": len(self._records),
                "unique_people": len(set(self._person)),
                "emb_dim": self.emb_dim,
            },
        )

    # ─────────────────────────────────────────────────────────────
    def persist(self):
        """Persist all records to pickle."""
        with timed(
            log,
            "persist_index",
            count=len(self._records),
            file=str(self.index_file),
        ):
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

        if embs.ndim == 1:
            # Allow single embedding vector
            embs = embs.reshape(1, -1)

        if embs.ndim != 2:
            raise ValueError(
                f"add_embeddings expected 2D array, got shape {embs.shape} "
                f"for person_id={person_id}"
            )

        if embs.shape[1] != self.emb_dim:
            raise ValueError(
                f"add_embeddings received dim={embs.shape[1]} but store emb_dim={self.emb_dim} "
                f"for person_id={person_id}. "
                f"Check that config.EMB_DIM ({self.emb_dim}) matches your embedder model "
                f"output dimension."
            )

        now = time.time()
        for e in embs:
            self._records.append(EmbRecord(person_id, e, now, source))

        log.info(
            "add_embs",
            extra={
                "person_id": person_id,
                "count": int(embs.shape[0]),
                "emb_dim": self.emb_dim,
            },
        )
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

        # Ensure q is 2D (n_queries, emb_dim)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.ndim != 2:
            raise ValueError(
                f"search_cosine expected 2D query array, got shape {q.shape}"
            )

        if q.shape[1] != self.emb_dim:
            raise ValueError(
                f"search_cosine query dim={q.shape[1]} but store emb_dim={self.emb_dim}. "
                f"This usually means the embedder model changed without updating "
                f"config.EMB_DIM or recreating the index."
            )

        if self._matrix.shape[1] != self.emb_dim:
            raise ValueError(
                f"search_cosine index matrix dim={self._matrix.shape[1]} "
                f"but store emb_dim={self.emb_dim}. The index file may be corrupted "
                f"or from an older model; delete {self.index_file} and re-enroll."
            )

        with timed(
            log,
            "search_cosine",
            queries=int(q.shape[0]),
            db_size=len(self._records),
        ):
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
                "emb_dim": self.emb_dim,
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
