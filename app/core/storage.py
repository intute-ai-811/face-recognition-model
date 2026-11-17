# app/core/storage.py
import os
import pickle
import shutil
import time
import threading
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from app import config
from app.logging_config import get_logger, timed

log = get_logger(__name__)

# Number of previous generations kept next to each pickle (file.pkl.1, file.pkl.2, ...).
BACKUP_GENERATIONS = 2


@dataclass
class EmbRecord:
    person_id: str
    embedding: np.ndarray
    ts: float
    source: str


@dataclass(frozen=True)
class _MatchSnapshot:
    """Immutable view used by search. Swapped atomically; never mutated in place."""
    db_norm: np.ndarray          # (N, D) L2-normalised embeddings
    person_ids: tuple[str, ...]  # length N, row-aligned with db_norm


def _generation_path(path: Path, n: int) -> Path:
    return path.with_name(f"{path.name}.{n}")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_pickle_dump(obj, path: Path, generations: int = BACKUP_GENERATIONS) -> None:
    """
    Crash-safe pickle write.

    1. Serialise to <path>.tmp and fsync it.
    2. Rotate previous generations (<path>.1 -> <path>.2, ...) and copy the
       current file to <path>.1. The live file is never removed in this step.
    3. os.replace(<path>.tmp, <path>) - atomic on POSIX, so readers and a crash
       at any point see either the old or the new complete file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")

    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())

    if generations > 0 and path.exists():
        for n in range(generations, 1, -1):
            older = _generation_path(path, n - 1)
            if older.exists():
                os.replace(older, _generation_path(path, n))
        gen1_tmp = path.with_name(f"{path.name}.1.tmp")
        shutil.copyfile(path, gen1_tmp)
        os.replace(gen1_tmp, _generation_path(path, 1))

    os.replace(tmp, path)
    _fsync_dir(path.parent)


def load_pickle_with_fallback(path: Path, generations: int = BACKUP_GENERATIONS):
    """
    Load <path>. If it exists but cannot be unpickled (truncated or corrupt),
    fall back to the newest readable generation (<path>.1, <path>.2, ...).

    Returns (data, loaded_from_path), or (None, None) when <path> does not exist.
    A missing primary file means "no data yet"; generations are only used to
    recover from a corrupt primary, never to resurrect a deliberately removed one.
    Raises the original error if the primary and every generation are unreadable.
    """
    path = Path(path)
    if not path.exists():
        return None, None

    try:
        with open(path, "rb") as f:
            return pickle.load(f), path
    except Exception as primary_err:
        log.error(
            "pickle_load_failed_trying_backups",
            extra={"file": str(path), "error": repr(primary_err)},
        )
        for n in range(1, generations + 1):
            gen = _generation_path(path, n)
            if not gen.exists():
                continue
            try:
                with open(gen, "rb") as f:
                    data = pickle.load(f)
                log.error(
                    "pickle_recovered_from_backup",
                    extra={"file": str(path), "backup": str(gen)},
                )
                return data, gen
            except Exception as gen_err:
                log.error(
                    "pickle_backup_unreadable",
                    extra={"backup": str(gen), "error": repr(gen_err)},
                )
        raise primary_err


class LocalStore:
    def __init__(self):
        if not hasattr(config, "EMB_DIM"):
            raise ValueError("config.EMB_DIM is not set.")

        self.emb_dim: int = int(config.EMB_DIM)
        # Re-entrant: persist() takes the lock and is also called while it is held.
        self._lock = threading.RLock()

        config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.index_file = config.EMBED_INDEX_FILE
        self.people_file: Path = getattr(config, "PEOPLE_FILE", config.DATA_DIR / "people.pkl")

        self._people: dict[str, dict] = {}
        self._load_people()

        self._records: list[EmbRecord] = []
        self._snapshot = _MatchSnapshot(np.zeros((0, self.emb_dim), np.float32), ())
        self._load()

    # ─────────────────────────────────────────────────────
    # People metadata
    # ─────────────────────────────────────────────────────

    def _load_people(self):
        try:
            data, src = load_pickle_with_fallback(self.people_file)
        except Exception as e:
            # Preserve previous behaviour (start with no names) but never silently:
            # the corrupt file and its generations stay on disk for inspection.
            log.error("people_load_failed", extra={"file": str(self.people_file), "error": str(e)})
            self._people = {}
            return

        if src is None:
            self._people = {}
            log.info("people_file_missing", extra={"file": str(self.people_file)})
            return

        self._people = data if isinstance(data, dict) else {}
        log.info("people_loaded", extra={"count": len(self._people), "file": str(src)})

    def _persist_people(self):
        with self._lock:
            with timed(log, "persist_people", count=len(self._people), file=str(self.people_file)):
                atomic_pickle_dump(self._people, self.people_file)
        log.info("people_persist_done", extra={"count": len(self._people)})

    def set_person(self, person_id: str, name: str | None):
        pid = str(person_id).strip()
        if not pid:
            return
        nm = (name or "").strip()
        if not nm:
            return

        with self._lock:
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

    # ─────────────────────────────────────────────────────
    # Embeddings index
    # ─────────────────────────────────────────────────────

    def _load(self):
        with self._lock:
            data, src = load_pickle_with_fallback(self.index_file)
            if src is not None:
                records = []
                bad_dims = set()

                for r in data:
                    emb = np.asarray(r["embedding"], np.float32)
                    if emb.ndim != 1:
                        bad_dims.add(tuple(emb.shape))
                    elif emb.shape[0] != self.emb_dim:
                        bad_dims.add(emb.shape[0])
                    else:
                        records.append(EmbRecord(r["person_id"], emb, r["ts"], r["source"]))

                if bad_dims:
                    raise ValueError(
                        f"Loaded embeddings with dimensions {bad_dims}, "
                        f"expected emb_dim={self.emb_dim}. Back up {self.index_file} "
                        f"and its .1/.2 generations, then re-enroll."
                    )

                self._records = records
                log.info(
                    "embeddings_loaded",
                    extra={"count": len(self._records), "file": str(src), "emb_dim": self.emb_dim},
                )
            else:
                self._records = []
                log.info("no_existing_index", extra={"file": str(self.index_file)})

            self._rebuild_cache()

    def _rebuild_cache(self):
        """Build a new immutable match snapshot from the current records. Caller holds the lock."""
        with self._lock:
            records = list(self._records)
            for r in records:
                if r.embedding.ndim != 1 or r.embedding.shape[0] != self.emb_dim:
                    raise ValueError(
                        f"Record for person_id={r.person_id} has embedding shape "
                        f"{r.embedding.shape}, expected ({self.emb_dim},). "
                        f"Back up {self.index_file} and re-enroll."
                    )

            if records:
                matrix = np.vstack([r.embedding for r in records])
                db_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
            else:
                db_norm = np.zeros((0, self.emb_dim), np.float32)
            db_norm.setflags(write=False)

            # Single reference assignment: searches see either the old or the new snapshot.
            self._snapshot = _MatchSnapshot(db_norm, tuple(r.person_id for r in records))

        log.info(
            "cache_rebuilt",
            extra={
                "records": len(records),
                "unique_people": len(set(self._snapshot.person_ids)),
                "emb_dim": self.emb_dim,
            },
        )

    def persist(self):
        with self._lock:
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
                atomic_pickle_dump(serializable, self.index_file)
            count = len(self._records)
        log.info("persist_done", extra={"count": count})

    def add_embeddings(self, person_id: str, embs: np.ndarray, source: str) -> int:
        if embs is None or embs.size == 0:
            log.warning("add_embs_empty", extra={"person_id": person_id})
            return 0

        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.ndim != 2:
            raise ValueError(f"add_embeddings expected 2D array, got shape {embs.shape}")
        if embs.shape[1] != self.emb_dim:
            raise ValueError(
                f"add_embeddings received dim={embs.shape[1]} but store emb_dim={self.emb_dim}"
            )

        now = time.time()
        with self._lock:
            for e in embs:
                self._records.append(EmbRecord(person_id, e, now, source))
            self.persist()
            self._rebuild_cache()

        log.info(
            "add_embs",
            extra={"person_id": person_id, "count": int(embs.shape[0]), "emb_dim": self.emb_dim},
        )
        return int(embs.shape[0])

    def search_cosine(self, q: np.ndarray, top_k: int) -> list[list[dict]]:
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2:
            raise ValueError(f"search_cosine expected 2D query, got shape {q.shape}")
        if q.shape[1] != self.emb_dim:
            raise ValueError(f"search_cosine query dim={q.shape[1]} but store emb_dim={self.emb_dim}")

        # Read the snapshot reference exactly once so matrix rows and ids always belong together.
        snap = self._snapshot
        if len(snap.person_ids) == 0:
            log.info("search_empty")
            return [[] for _ in range(q.shape[0])]

        with timed(log, "search_cosine", queries=int(q.shape[0]), db_size=len(snap.person_ids)):
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims = q_norm @ snap.db_norm.T

            results = []
            for i in range(q.shape[0]):
                agg = {}
                for j, pid in enumerate(snap.person_ids):
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
                "unique_people": len(set(snap.person_ids)),
                "top_k": top_k,
            },
        )
        return results

    def people(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pid in self._snapshot.person_ids:
            counts[pid] = counts.get(pid, 0) + 1
        log.info("people_summary", extra={"unique_people": len(counts)})
        return counts

    def delete_person(self, person_id: str) -> int:
        pid = str(person_id).strip()
        if not pid:
            return 0

        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.person_id != pid]
            if pid in self._people:
                del self._people[pid]
                self._persist_people()
            self.persist()
            self._rebuild_cache()
            removed = before - len(self._records)

        log.info("delete_person", extra={"person_id": pid, "removed_embeddings": removed})
        return int(removed)

    def save_aligned(self, person_id: str, aligned_img) -> None:
        if aligned_img is None:
            log.warning("save_aligned_none", extra={"person_id": person_id})
            return
        try:
            import cv2
            save_dir = config.FACES_DIR / str(person_id)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(save_dir / fname), aligned_img)
        except Exception as e:
            log.warning("save_aligned_failed", extra={"person_id": person_id, "error": str(e)})

    def reload(self) -> None:
        log.info("store_reload")
        with self._lock:
            self._load_people()
            self._load()
