"""
One-way, non-destructive migration: pickle enrollments -> SQLite.

    python scripts/migrate_pickle_to_sqlite.py --data-dir data --dry-run
    python scripts/migrate_pickle_to_sqlite.py --data-dir data

Guarantees:
  - index.pkl and people.pkl are only read, never written (their SHA-256 is
    checked before and after).
  - An existing database is never overwritten; the script aborts instead.
  - The database is built under a temporary name and only renamed to
    attendance.db after verification passes, so a failed or interrupted run
    leaves no database behind.
  - Verification: same people and names, same embedding count per person,
    every vector byte-identical, and identical search results (person ids and
    scores) for a probe set under both stores.

The pickles are loaded with pickle.load. Run this only on your own data
directory (trusted operator action), never on files from elsewhere.

Inside the production container (the data volume is mounted at /code/data):
    docker exec face-recognition-api python scripts/migrate_pickle_to_sqlite.py --data-dir /code/data --dry-run
Full procedure: docs/operations/sqlite-migration.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MigrationError(RuntimeError):
    pass


def _point_config_at(data_dir: Path) -> None:
    from app import config
    config.DATA_DIR = data_dir
    config.EMBED_DIR = data_dir / "embeddings"
    config.EMBED_INDEX_FILE = data_dir / "embeddings" / "index.pkl"
    config.PEOPLE_FILE = data_dir / "people.pkl"
    config.FACES_DIR = data_dir / "faces"
    config.DB_FILE = data_dir / "attendance.db"


def migrate(data_dir: Path, db_path: Path, model_path: Path, dry_run: bool = False, probes: int = 200) -> dict:
    from app import config
    from app.core.sqlite_store import SqliteStore, connect, file_sha256, init_schema, model_id_for

    data_dir = Path(data_dir)
    db_path = Path(db_path)
    index_file = data_dir / "embeddings" / "index.pkl"
    people_file = data_dir / "people.pkl"
    emb_dim = int(config.EMB_DIM)

    # ── preflight ────────────────────────────────────────────────────────────
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            raise MigrationError(f"{p} already exists - refusing to overwrite. Move it away first if you mean to re-run.")
    if not index_file.exists():
        raise MigrationError(f"{index_file} not found - nothing to migrate")
    if not Path(model_path).exists():
        raise MigrationError(f"model {model_path} not found (needed to tag embeddings with the model that made them)")

    hashes_before = {str(f): file_sha256(f) for f in (index_file, people_file) if f.exists()}

    try:
        with open(index_file, "rb") as f:
            records = pickle.load(f)
    except Exception as e:
        raise MigrationError(
            f"cannot read {index_file}: {e!r}. Restore it from a backup (or index.pkl.1) before migrating."
        ) from e
    people = {}
    if people_file.exists():
        try:
            with open(people_file, "rb") as f:
                people = pickle.load(f)
        except Exception as e:
            raise MigrationError(f"cannot read {people_file}: {e!r}") from e
        if not isinstance(people, dict):
            raise MigrationError(f"{people_file} does not contain a dict")

    vectors, rows = [], []
    for i, r in enumerate(records):
        vec = np.asarray(r["embedding"], np.float32)
        if vec.shape != (emb_dim,):
            raise MigrationError(f"record {i} ({r.get('person_id')}) has shape {vec.shape}, expected ({emb_dim},)")
        if not np.isfinite(vec).all():
            raise MigrationError(f"record {i} ({r.get('person_id')}) contains NaN/inf")
        vectors.append(vec)
        rows.append((str(r["person_id"]), float(r.get("ts") or time.time()), r.get("source")))

    model_sha = file_sha256(Path(model_path))
    model_id = model_id_for(Path(model_path), model_sha)

    # ── build under a temporary name ─────────────────────────────────────────
    tmp_db = db_path.with_name(db_path.name + ".migrating")
    for suffix in ("", "-wal", "-shm"):
        Path(str(tmp_db) + suffix).unlink(missing_ok=True)

    now = time.time()
    conn = connect(tmp_db)
    try:
        init_schema(conn)
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO models(model_id, sha256, emb_dim, threshold, status, created_at) VALUES (?,?,?,?,?,?)",
                (model_id, model_sha, emb_dim, config.SIMILARITY_THRESHOLD, "active", now),
            )
            all_pids = {pid for pid, _, _ in rows} | {str(k) for k in people}
            for pid in sorted(all_pids):
                rec = people.get(pid) or {}
                name = (rec.get("name") or None) if isinstance(rec, dict) else None
                ts = float(rec.get("updated_at") or now) if isinstance(rec, dict) else now
                conn.execute(
                    "INSERT INTO people(person_id, name, created_at, updated_at) VALUES (?,?,?,?)",
                    (pid, name, ts, ts),
                )
            conn.executemany(
                "INSERT INTO embeddings(person_id, model_id, vector, source, created_at) VALUES (?,?,?,?,?)",
                [(pid, model_id, vec.tobytes(), src, ts) for (pid, ts, src), vec in zip(rows, vectors)],
            )
            conn.execute(
                "INSERT INTO audit_log(ts, actor, action, person_id, detail) VALUES (?,?,?,?,?)",
                (now, "migration", "migrated_from_pickle", None,
                 json.dumps({"embeddings": len(rows), "people": len(all_pids), "source_sha256": hashes_before})),
            )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    # ── verify against the pickle store ──────────────────────────────────────
    _point_config_at(data_dir)
    from app.core.storage import LocalStore

    ref = LocalStore()
    new = SqliteStore(tmp_db, model_id=model_id, model_sha256=model_sha, emb_dim=emb_dim, mirror=None)
    try:
        problems = []
        if ref.people() != new.people():
            problems.append("embedding counts per person differ")
        for pid in all_pids:
            if ref.get_person_name(pid) != new.get_person_name(pid):
                problems.append(f"name differs for person {pid}")
                break
        if len(new._vecs) != len(vectors) or any(a.tobytes() != b.tobytes() for a, b in zip(new._vecs, vectors)):
            problems.append("stored vectors are not byte-identical")

        rng = np.random.default_rng(0)
        probe_vecs = []
        if vectors:
            idx = rng.choice(len(vectors), size=min(probes, len(vectors)), replace=False)
            probe_vecs += [vectors[i] for i in idx]
        probe_vecs += list(rng.standard_normal((min(probes, 50), emb_dim)).astype(np.float32))
        mismatched = 0
        if probe_vecs:
            q = np.stack(probe_vecs)
            for a, b in zip(ref.search_cosine(q, config.TOP_K), new.search_cosine(q, config.TOP_K)):
                if a != b:
                    mismatched += 1
        if mismatched:
            problems.append(f"{mismatched}/{len(probe_vecs)} probe searches differ")
    finally:
        new.close()

    hashes_after = {str(f): file_sha256(Path(f)) for f in hashes_before}
    if hashes_after != hashes_before:
        problems.append("pickle files changed during migration")

    report = {
        "data_dir": str(data_dir),
        "database": str(db_path),
        "model_id": model_id,
        "people": len(all_pids),
        "people_with_names": sum(1 for p in all_pids if ref.get_person_name(p)),
        "embeddings": len(rows),
        "probe_searches_compared": len(probe_vecs),
        "pickle_sha256": hashes_before,
        "problems": problems,
        "dry_run": dry_run,
    }

    if problems:
        for suffix in ("", "-wal", "-shm"):
            Path(str(tmp_db) + suffix).unlink(missing_ok=True)
        raise MigrationError("verification failed: " + "; ".join(problems))

    if dry_run:
        for suffix in ("", "-wal", "-shm"):
            Path(str(tmp_db) + suffix).unlink(missing_ok=True)
        return report

    for suffix in ("-wal", "-shm"):
        Path(str(tmp_db) + suffix).unlink(missing_ok=True)
    fd = os.open(str(tmp_db), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_db, db_path)
    # Sanity: the final file opens and has the right row count.
    check = sqlite3.connect(str(db_path))
    try:
        n = check.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    finally:
        check.close()
    if n != len(rows):
        raise MigrationError(f"final database has {n} embeddings, expected {len(rows)}")
    return report


def main() -> None:
    from app import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(config.DATA_DIR))
    ap.add_argument("--db", type=Path, help="default: <data-dir>/attendance.db")
    ap.add_argument("--model", type=Path, default=Path(config.MOBILEFACENET_MODEL))
    ap.add_argument("--dry-run", action="store_true", help="build and verify in a temp file, then delete it")
    ap.add_argument("--probes", type=int, default=200)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    db = args.db or (args.data_dir / "attendance.db")
    try:
        report = migrate(args.data_dir, db, args.model, dry_run=args.dry_run, probes=args.probes)
    except MigrationError as e:
        print(f"MIGRATION ABORTED: {e}", file=sys.stderr)
        print("Nothing was changed.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, indent=1))
    if args.dry_run:
        print("\nDry run OK - verification passed; no database was kept.")
    else:
        print(f"\nMigration OK: {db}\nNext: set STORAGE_BACKEND=sqlite in app.env and restart the container.")


if __name__ == "__main__":
    main()
