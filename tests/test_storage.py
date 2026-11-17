"""Regression tests for crash-safe persistence and race-free matching (PR 1)."""
import pickle
import threading
import time

import numpy as np
import pytest

from app import config
import app.core.storage as storage
from app.core.storage import LocalStore, atomic_pickle_dump, load_pickle_with_fallback

DIM = config.EMB_DIM


@pytest.fixture
def store_paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    embed = data / "embeddings"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "EMBED_DIR", embed)
    monkeypatch.setattr(config, "EMBED_INDEX_FILE", embed / "index.pkl")
    monkeypatch.setattr(config, "PEOPLE_FILE", data / "people.pkl")
    monkeypatch.setattr(config, "FACES_DIR", data / "faces")
    return embed / "index.pkl", data / "people.pkl"


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


# ── atomic writes ────────────────────────────────────────────────────────────

def test_roundtrip_persists_and_reloads(store_paths):
    s = LocalStore()
    s.set_person("70", "Alice")
    s.add_embeddings("70", np.stack([unit(0), unit(1)]), "a.jpg")

    s2 = LocalStore()
    assert s2.people() == {"70": 2}
    assert s2.get_person_name("70") == "Alice"
    res = s2.search_cosine(unit(0), top_k=3)
    assert res[0][0]["person_id"] == "70"
    assert res[0][0]["score"] == pytest.approx(1.0)


def test_file_format_is_unchanged_plain_list_of_dicts(store_paths):
    index_file, _ = store_paths
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list)
    assert set(data[0]) == {"person_id", "embedding", "ts", "source"}


def test_crash_during_write_leaves_previous_index_intact(store_paths, monkeypatch):
    index_file, _ = store_paths
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    good_bytes = index_file.read_bytes()

    real_dump = pickle.dump

    def crashing_dump(obj, f, *a, **kw):
        f.write(b"\x80\x04partial")  # half-written payload
        raise OSError("simulated crash / disk full")

    monkeypatch.setattr(storage.pickle, "dump", crashing_dump)
    with pytest.raises(OSError):
        s.add_embeddings("71", unit(1), "b.jpg")
    monkeypatch.setattr(storage.pickle, "dump", real_dump)

    # The live file was never touched, so a restart loads the last good state.
    assert index_file.read_bytes() == good_bytes
    assert LocalStore().people() == {"70": 1}


def test_truncated_primary_falls_back_to_newest_generation(store_paths):
    index_file, _ = store_paths
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    s.add_embeddings("71", unit(1), "b.jpg")  # index.pkl.1 now holds {70}
    index_file.write_bytes(index_file.read_bytes()[:10])  # simulate torn write by old code

    recovered = LocalStore()
    assert recovered.people() == {"70": 1}


def test_generations_rotate_and_are_capped(store_paths):
    index_file, _ = store_paths
    s = LocalStore()
    for i in range(5):
        s.add_embeddings(str(i), unit(i), f"{i}.jpg")
    names = sorted(p.name for p in index_file.parent.iterdir())
    assert names == ["index.pkl", "index.pkl.1", "index.pkl.2"]
    with open(index_file.with_name("index.pkl.1"), "rb") as f:
        assert len(pickle.load(f)) == 4


def test_missing_primary_starts_empty_and_does_not_resurrect_backups(store_paths):
    index_file, _ = store_paths
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    s.add_embeddings("71", unit(1), "b.jpg")
    index_file.unlink()  # operator deliberately removed the index
    assert LocalStore().people() == {}


def test_all_generations_corrupt_raises(store_paths):
    index_file, _ = store_paths
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    s.add_embeddings("71", unit(1), "b.jpg")
    for p in index_file.parent.glob("index.pkl*"):
        p.write_bytes(b"garbage")
    with pytest.raises(Exception):
        LocalStore()


def test_corrupt_people_file_recovers_names_from_backup(store_paths):
    _, people_file = store_paths
    s = LocalStore()
    s.set_person("70", "Alice")
    s.set_person("71", "Bob")
    people_file.write_bytes(b"\x80")
    s2 = LocalStore()
    assert s2.get_person_name("70") == "Alice"


def test_load_pickle_with_fallback_missing_returns_none(tmp_path):
    assert load_pickle_with_fallback(tmp_path / "nope.pkl") == (None, None)


def test_atomic_dump_leaves_no_tmp_files(tmp_path):
    p = tmp_path / "x.pkl"
    atomic_pickle_dump({"a": 1}, p)
    atomic_pickle_dump({"a": 2}, p)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.pkl", "x.pkl.1"]


# ── empty store ──────────────────────────────────────────────────────────────

def test_empty_store_search_returns_one_empty_list_per_query(store_paths):
    s = LocalStore()
    assert s.search_cosine(np.stack([unit(0), unit(1)]), top_k=3) == [[], []]


def test_delete_last_person_then_search_is_empty(store_paths):
    s = LocalStore()
    s.add_embeddings("70", unit(0), "a.jpg")
    assert s.delete_person("70") == 1
    assert s.search_cosine(unit(0), top_k=3) == [[]]


# ── matching unchanged + race-free ──────────────────────────────────────────

def test_search_scores_match_reference_formula(store_paths):
    rng = np.random.default_rng(0)
    s = LocalStore()
    embs = rng.standard_normal((6, DIM)).astype(np.float32)
    pids = ["a", "a", "b", "b", "c", "c"]
    for pid, e in zip(pids, embs):
        s.add_embeddings(pid, e, "x")
    q = rng.standard_normal((2, DIM)).astype(np.float32)

    # Reference = the pre-PR implementation's maths.
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    dn = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = qn @ dn.T
    for i, res in enumerate(s.search_cosine(q, top_k=3)):
        expected = {}
        for j, pid in enumerate(pids):
            expected[pid] = max(expected.get(pid, -9), float(sims[i, j]))
        ranked = sorted(expected.items(), key=lambda kv: -kv[1])
        assert [r["person_id"] for r in res] == [pid for pid, _ in ranked]
        assert [r["score"] for r in res] == [sc for _, sc in ranked]


def test_concurrent_enroll_delete_and_search_never_misattribute(store_paths):
    s = LocalStore()
    s.add_embeddings("A", unit(0), "a")
    stop = threading.Event()
    errors = []

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
                res = s.search_cosine(unit(0), top_k=1)[0]
                if not res or res[0]["person_id"] != "A" or res[0]["score"] < 0.999:
                    errors.append(res)
            except Exception as e:  # IndexError etc. under the old race
                errors.append(repr(e))

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join()
    assert errors == []
