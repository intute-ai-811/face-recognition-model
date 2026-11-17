"""SQLite storage backend and pickle -> SQLite migration (PR 7)."""
import importlib.util
import pickle
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.core.sqlite_store import SqliteStore
from app.core.storage import LocalStore

DIM = config.EMB_DIM
MODEL_ID, MODEL_SHA = "w600k_mbf:test", "a" * 64

_spec = importlib.util.spec_from_file_location(
    "migrate_pickle_to_sqlite", Path(__file__).resolve().parents[1] / "scripts" / "migrate_pickle_to_sqlite.py"
)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


def unit(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "EMBED_DIR", data / "embeddings")
    monkeypatch.setattr(config, "EMBED_INDEX_FILE", data / "embeddings" / "index.pkl")
    monkeypatch.setattr(config, "PEOPLE_FILE", data / "people.pkl")
    monkeypatch.setattr(config, "FACES_DIR", data / "faces")
    monkeypatch.setattr(config, "DB_FILE", data / "attendance.db")
    model = tmp_path / "w600k_mbf.onnx"
    model.write_bytes(b"fake model bytes")
    monkeypatch.setattr(config, "MOBILEFACENET_MODEL", model)
    return data, model


def store(data, mirror=None, model_id=MODEL_ID, sha=MODEL_SHA):
    return SqliteStore(data / "attendance.db", model_id=model_id, model_sha256=sha, mirror=mirror)


# ── store behaviour ──────────────────────────────────────────────────────────

def test_roundtrip(paths):
    data, _ = paths
    s = store(data)
    s.set_person("70", "Alice")
    s.add_embeddings("70", np.stack([unit(0), unit(1)]), "a.jpg")
    s.add_embeddings("71", unit(2), "b.jpg")
    s.close()

    s2 = store(data)
    assert s2.people() == {"70": 2, "71": 1}
    assert s2.get_person_name("70") == "Alice"
    assert s2.get_person_name("71") is None
    assert s2.search_cosine(unit(2), 3)[0][0] == {"person_id": "71", "score": pytest.approx(1.0)}


def test_search_identical_to_pickle_store(paths):
    data, _ = paths
    rng = np.random.default_rng(1)
    embs = rng.standard_normal((12, DIM)).astype(np.float32)
    pids = [str(i % 4) for i in range(12)]
    a, b = LocalStore(), store(data)
    for pid, e in zip(pids, embs):
        a.add_embeddings(pid, e, "x")
        b.add_embeddings(pid, e, "x")
    q = rng.standard_normal((5, DIM)).astype(np.float32)
    assert a.search_cosine(q, 3) == b.search_cosine(q, 3)
    assert a.people() == b.people()


def test_empty_store(paths):
    data, _ = paths
    assert store(data).search_cosine(np.stack([unit(0), unit(1)]), 3) == [[], []]


def test_delete_cascades_and_is_audited(paths):
    data, _ = paths
    s = store(data)
    s.set_person("70", "Alice")
    s.add_embeddings("70", np.stack([unit(0), unit(1)]), "a")
    assert s.delete_person("70") == 2
    assert s.people() == {} and s.get_person_name("70") is None
    conn = sqlite3.connect(data / "attendance.db")
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    actions = [r[0] for r in conn.execute("SELECT action FROM audit_log ORDER BY id")]
    assert actions == ["set_name", "enroll", "delete"]
    # audit log holds no vectors
    assert all(r[0] is None or b"\x00" not in r[0].encode() for r in conn.execute("SELECT detail FROM audit_log"))


def test_other_model_embeddings_are_never_loaded(paths):
    data, _ = paths
    old = store(data, model_id="old_model:1", sha="b" * 64)
    old.add_embeddings("70", unit(0), "a")
    old.close()
    new = store(data)  # different active model
    assert new.people() == {}
    assert new.search_cosine(unit(0), 3) == [[]]
    new.add_embeddings("70", unit(5), "b")
    new.close()
    again = store(data, model_id="old_model:1", sha="b" * 64)
    assert again.people() == {"70": 1}
    assert again.search_cosine(unit(0), 1)[0][0]["score"] == pytest.approx(1.0)


def test_failed_write_rolls_back_and_keeps_memory_consistent(paths, monkeypatch):
    data, _ = paths
    s = store(data)
    s.add_embeddings("70", unit(0), "a")

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(s, "_audit", boom)
    with pytest.raises(sqlite3.OperationalError):
        s.add_embeddings("71", unit(1), "b")
    assert s.people() == {"70": 1}
    monkeypatch.undo()
    assert store(data).people() == {"70": 1}


def test_concurrent_enroll_delete_and_search_never_misattribute(paths):
    data, _ = paths
    s = store(data)
    s.add_embeddings("A", unit(0), "a")
    stop, errors = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            pid = f"tmp{i % 3}"
            s.add_embeddings(pid, unit(10 + i % 3), "w")
            s.delete_person(pid)
            i += 1

    def reader():
        while not stop.is_set():
            try:
                res = s.search_cosine(unit(0), 1)[0]
                if not res or res[0]["person_id"] != "A" or res[0]["score"] < 0.999:
                    errors.append(res)
            except Exception as e:
                errors.append(repr(e))

    ts = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in ts:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in ts:
        t.join()
    assert errors == []


# ── pickle mirror (rollback safety) ──────────────────────────────────────────

def test_pickle_mirror_receives_every_write(paths):
    data, _ = paths
    s = store(data, mirror=LocalStore())
    s.set_person("70", "Alice")
    s.add_embeddings("70", unit(0), "a")
    s.add_embeddings("71", unit(1), "b")
    s.delete_person("71")
    back = LocalStore()  # what a rollback to STORAGE_BACKEND=pickle would load
    assert back.people() == {"70": 1}
    assert back.get_person_name("70") == "Alice"


def test_mirror_failure_does_not_fail_the_request(paths):
    data, _ = paths

    class BrokenMirror:
        def add_embeddings(self, *a):
            raise OSError("disk full")

    s = store(data, mirror=BrokenMirror())
    assert s.add_embeddings("70", unit(0), "a") == 1
    assert s.people() == {"70": 1}


# ── factory ──────────────────────────────────────────────────────────────────

def test_factory_default_is_pickle(paths, monkeypatch):
    from app.core.store_factory import create_store
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    assert isinstance(create_store(), LocalStore)


def test_factory_refuses_sqlite_without_migration_when_pickles_exist(paths, monkeypatch):
    from app.core.store_factory import create_store
    LocalStore().add_embeddings("70", unit(0), "a")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    with pytest.raises(RuntimeError, match="migrate_pickle_to_sqlite"):
        create_store()


def test_factory_sqlite_fresh_install(paths, monkeypatch):
    from app.core.store_factory import create_store
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("STORAGE_MIRROR_PICKLE", "0")
    s = create_store()
    assert isinstance(s, SqliteStore) and s.people() == {} and s.mirror is None


def test_factory_rejects_unknown_backend(paths, monkeypatch):
    from app.core.store_factory import create_store
    monkeypatch.setenv("STORAGE_BACKEND", "mongo")
    with pytest.raises(ValueError):
        create_store()


# ── migration ────────────────────────────────────────────────────────────────

def _seed_pickles(n_people=6, per=3, seed=3):
    rng = np.random.default_rng(seed)
    ls = LocalStore()
    for p in range(n_people):
        if p % 2 == 0:
            ls.set_person(str(70 + p), f"Person {p}")
        ls.add_embeddings(str(70 + p), rng.standard_normal((per, DIM)).astype(np.float32), f"{p}.jpg")
    return ls


def test_migration_happy_path_then_service_loads_same_data(paths, monkeypatch):
    data, model = paths
    ref = _seed_pickles()
    before = {f: f.read_bytes() for f in (data / "embeddings" / "index.pkl", data / "people.pkl")}

    report = mig.migrate(data, data / "attendance.db", model)
    assert report["problems"] == [] and report["embeddings"] == 18 and report["people"] == 6
    assert report["people_with_names"] == 3
    assert {f: f.read_bytes() for f in before} == before  # pickles untouched
    assert not (data / "attendance.db.migrating").exists()

    from app.core.store_factory import create_store
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    s = create_store()
    q = np.random.default_rng(9).standard_normal((4, DIM)).astype(np.float32)
    assert s.search_cosine(q, 3) == ref.search_cosine(q, 3)
    assert s.people() == ref.people()
    assert s.get_person_name("70") == "Person 0" and s.get_person_name("71") is None


def test_migration_dry_run_keeps_no_database(paths):
    data, model = paths
    _seed_pickles()
    report = mig.migrate(data, data / "attendance.db", model, dry_run=True)
    assert report["dry_run"] is True and report["problems"] == []
    assert not (data / "attendance.db").exists()
    assert not list(data.glob("attendance.db*"))


def test_migration_refuses_to_overwrite_existing_db(paths):
    data, model = paths
    _seed_pickles()
    (data / "attendance.db").write_bytes(b"precious")
    with pytest.raises(mig.MigrationError, match="refusing to overwrite"):
        mig.migrate(data, data / "attendance.db", model)
    assert (data / "attendance.db").read_bytes() == b"precious"


def test_migration_aborts_on_corrupt_pickle_and_leaves_nothing(paths):
    data, model = paths
    _seed_pickles()
    (data / "embeddings" / "index.pkl").write_bytes(b"\x80\x04trunc")
    with pytest.raises(mig.MigrationError, match="cannot read"):
        mig.migrate(data, data / "attendance.db", model)
    assert not list(data.glob("attendance.db*"))


def test_migration_aborts_on_bad_vector(paths):
    data, model = paths
    (data / "embeddings").mkdir(parents=True)
    with open(data / "embeddings" / "index.pkl", "wb") as f:
        pickle.dump([{"person_id": "70", "embedding": np.full(DIM, np.nan, np.float32), "ts": 0.0, "source": "x"}], f)
    with pytest.raises(mig.MigrationError, match="NaN"):
        mig.migrate(data, data / "attendance.db", model)
    assert not list(data.glob("attendance.db*"))


def test_migration_verification_failure_leaves_no_database(paths, monkeypatch):
    data, model = paths
    _seed_pickles()
    real = SqliteStore.search_cosine

    def tampered(self, q, top_k):
        res = real(self, q, top_k)
        if res and res[0]:
            res[0][0] = {**res[0][0], "score": res[0][0]["score"] - 0.1}
        return res

    monkeypatch.setattr(SqliteStore, "search_cosine", tampered)
    with pytest.raises(mig.MigrationError, match="probe searches differ"):
        mig.migrate(data, data / "attendance.db", model)
    assert not list(data.glob("attendance.db*"))
