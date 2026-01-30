# app/core/storage.py
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app import config
from app.logging_config import get_logger, timed

log = get_logger(__name__)


@dataclass
class EmbRecord:
    person_id: str
    embedding: np.ndarray
    ts: float
    source: str


class LocalStore:
    """
    Handles local storage of embeddings (index.pkl) and person metadata (people.pkl).
    Embeddings are stored as pickle (index.pkl). Person metadata (e.g. name) is
    stored separately in people.pkl so you can return person_name without ERP mapping.
    """

    def __init__(self):
        if not hasattr(config, "EMB_DIM"):
            raise ValueError("config.EMB_DIM is not set. Please define it to match embedder output.")

        self.emb_dim: int = int(config.EMB_DIM)

        config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.index_file = config.EMBED_INDEX_FILE

        # Person metadata file (id -> {"name": "...", "updated_at": ...})
        self.people_file: Path = getattr(config, "PEOPLE_FILE", config.DATA_DIR / "people.pkl")
        self._people: dict[str, dict] = {}
        self._load_people()

        self._records: list[EmbRecord] = []
        self._matrix: np.ndarray | None = None
        self._person: list[str] = []
        self._load()

    # ───────────────────────────────
    # People metadata (id -> name)
    # ───────────────────────────────
    def _load_people(self):
        if self.people_file.exists():
            try:
                with open(self.people_file, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self._people = data
                else:
                    self._people = {}
                log.info("people_loaded", extra={"count": len(self._people), "file": str(self.people_file)})
            except Exception as e:
                log.warning("people_load_failed", extra={"file": str(self.people_file), "error": str(e)})
                self._people = {}
        else:
            self._people = {}
            log.info("people_file_missing", extra={"file": str(self.people_file)})

    def _persist_people(self):
        self.people_file.parent.mkdir(parents=True, exist_ok=True)
        with timed(log, "persist_people", count=len(self._people), file=str(self.people_file)):
            with open(self.people_file, "wb") as f:
                pickle.dump(self._people, f)
        log.info("people_persist_done", extra={"count": len(self._people)})

    def set_person(self, person_id: str, name: str | None):
        pid = str(person_id).strip()
        if not pid:
            return
        nm = (name or "").strip()
        if not nm:
            return  # don't overwrite with empty

        self._people[pid] = {"name": nm, "updated_at": time.time()}
        self._persist_people()

    def get_person_name(self, person_id: str) -> str | None:
        pid = str(person_id).strip()
        if not pid:
            return None
        rec = self._people.get(pid)
        if not rec:
            return None
        return rec.get("name") or None

    # ───────────────────────────────
    # Embeddings index
    # ───────────────────────────────
    def _load(self):
        """Load existing embeddings from disk into memory and validate dimensions."""
        if self.index_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)

            self._records = []
            bad_dims = set()

            for r in data:
                emb = np.asarray(r["embedding"], np.float32)
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
                raise ValueError(
                    f"Loaded embeddings from {self.index_file} with dimensions {bad_dims}, "
                    f"but expected emb_dim={self.emb_dim}. Delete {self.index_file} and re-enroll."
                )

            log.info(
                "embeddings_loaded",
                extra={"count": len(self._records), "file": str(self.index_file), "emb_dim": self.emb_dim},
            )
        else:
            log.info("no_existing_index", extra={"file": str(self.index_file)})

        self._rebuild_cache()

    def _rebuild_cache(self):
        """Rebuild in-memory matrix and person list for cosine search."""
        if self._records:
            for r in self._records:
                if r.embedding.ndim != 1 or r.embedding.shape[0] != self.emb_dim:
                    raise ValueError(
                        f"Record for person_id={r.person_id} has embedding shape {r.embedding.shape}, "
                        f"expected ({self.emb_dim},). Delete {self.index_file} and re-enroll."
                    )

            self._matrix = np.vstack([r.embedding for r in self._records])
            self._person = [r.person_id for r in self._records]
        else:
            self._matrix = np.zeros((0, self.emb_dim), np.float32)
            self._person = []

        log.info(
            "cache_rebuilt",
            extra={"records": len(self._records), "unique_people": len(set(self._person)), "emb_dim": self.emb_dim},
        )

    def persist(self):
        """Persist all embedding records to pickle."""
        with timed(log, "persist_index", count=len(self._records), file=str(self.index_file)):
            serializable = [
                {"person_id": r.person_id, "embedding": r.embedding.astype(np.float32), "ts": r.ts, "source": r.source}
                for r in self._records
            ]
            with open(self.index_file, "wb") as f:
                pickle.dump(serializable, f)
        log.info("persist_done", extra={"count": len(self._records)})

    def add_embeddings(self, person_id: str, embs: np.ndarray, source: str) -> int:
        if embs is None or embs.size == 0:
            log.warning("add_embs_empty", extra={"person_id": person_id})
            return 0

        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        if embs.ndim != 2:
            raise ValueError(f"add_embeddings expected 2D array, got shape {embs.shape} for person_id={person_id}")

        if embs.shape[1] != self.emb_dim:
            raise ValueError(
                f"add_embeddings received dim={embs.shape[1]} but store emb_dim={self.emb_dim} for person_id={person_id}"
            )

        now = time.time()
        for e in embs:
            self._records.append(EmbRecord(person_id, e, now, source))

        log.info("add_embs", extra={"person_id": person_id, "count": int(embs.shape[0]), "emb_dim": self.emb_dim})
        self.persist()
        self._rebuild_cache()
        return int(embs.shape[0])

    def search_cosine(self, q: np.ndarray, top_k: int):
        if self._matrix is None or len(self._records) == 0:
            log.info("search_empty")
            return []

        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.ndim != 2:
            raise ValueError(f"search_cosine expected 2D query array, got shape {q.shape}")

        if q.shape[1] != self.emb_dim:
            raise ValueError(f"search_cosine query dim={q.shape[1]} but store emb_dim={self.emb_dim}.")

        with timed(log, "search_cosine", queries=int(q.shape[0]), db_size=len(self._records)):
            sims = q @ self._matrix.T

        results = []
        for i in range(q.shape[0]):
            agg = {}
            for j, pid in enumerate(self._person):
                s = float(sims[i, j])
                if (pid not in agg) or (s > agg[pid]):
                    agg[pid] = s
            ranked = sorted(({"person_id": pid, "score": sc} for pid, sc in agg.items()), key=lambda x: -x["score"])[:top_k]
            results.append(ranked)

        log.info("search_done", extra={"queries": int(q.shape[0]), "unique_people": len(set(self._person)), "top_k": top_k})
        return results

    def people(self) -> dict[str, int]:
        counts = {}
        for r in self._records:
            counts[r.person_id] = counts.get(r.person_id, 0) + 1
        log.info("people_summary", extra={"unique_people": len(counts)})
        return counts

    def delete_person(self, person_id: str) -> int:
        """Delete all embeddings and metadata for a person."""
        pid = str(person_id).strip()
        if not pid:
            return 0

        before = len(self._records)

        # Remove embeddings
        self._records = [r for r in self._records if r.person_id != pid]

        # Remove person metadata (name)
        if pid in self._people:
            del self._people[pid]
            self._persist_people()

        self.persist()
        self._rebuild_cache()

        removed = before - len(self._records)
        log.info("delete_person", extra={"person_id": pid, "removed_embeddings": removed})
        return int(removed)
