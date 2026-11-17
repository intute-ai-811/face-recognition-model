# Runbook: switch enrollment storage from pickle to SQLite

**Who:** the service owner, on the VPS. **When:** a quiet period with no enrollments. **Duration:** about 10 minutes.

It needs an image built from a commit that includes plan PR 7 (`app/core/sqlite_store.py`).

Nothing in this procedure modifies or deletes the existing pickle files. Every step can be undone by switching `STORAGE_BACKEND` back.

## What changes

| | Before | After |
| --- | --- | --- |
| Enrollment data | `data/embeddings/index.pkl`, `data/people.pkl` | `data/attendance.db` (SQLite, WAL), plus the pickles kept up to date as a rollback copy |
| Writes | whole file rewritten per change | one transaction per change |
| Model tagging | none | every embedding row records the model that produced it |
| Audit | none | `audit_log` table: enroll / rename / delete (no biometric data) |
| Recognition | max cosine per person, threshold 0.50 | identical code and results |

## Procedure

All commands run on the VPS from `~/face-recognition-model`.

1. **Back up** (in addition to the 2026-11-13 backup):

    ```bash
    cp -a data "data-backup-$(date +%F)-pre-sqlite"
    sha256sum data/embeddings/index.pkl data/people.pkl
    ```

2. **Dry run** inside the running container. It builds and verifies a database in a temporary file, then deletes it:

    ```bash
    docker exec face-recognition-api python scripts/migrate_pickle_to_sqlite.py --data-dir /code/data --dry-run
    ```

    Expect `"problems": []` and `Dry run OK`. Check that `people` and `embeddings` match `GET /admin/people`. If it aborts, stop here: nothing was changed.

3. **Migrate for real:**

    ```bash
    docker exec face-recognition-api python scripts/migrate_pickle_to_sqlite.py --data-dir /code/data
    ```

    This creates `data/attendance.db`. The service still runs on pickle, so nothing has switched yet.

4. **Switch the backend.** Add this line to `app.env`:

    ```text
    STORAGE_BACKEND=sqlite
    ```

    `STORAGE_MIRROR_PICKLE=1` is the default, so the pickles keep receiving every write.

5. **Restart with the new env.** An env-file change needs the container to be recreated, not just restarted. Run the **Rollback VPS deployment** workflow with the image tag that is currently deployed (`cat current_image`), or re-run the deploy workflow. Both recreate the container with the same image and the new `app.env`, then wait for `/healthz`.

6. **Verify:**

    ```bash
    docker logs face-recognition-api 2>&1 | grep -E "storage_backend|sqlite_store_loaded"
    ```

    Expect `backend=sqlite`, the same embedding count as before, and `other_model_embeddings_ignored=0`. Then mark attendance once at a kiosk.

## Rollback

Set `STORAGE_BACKEND=pickle` (or delete the line) in `app.env` and recreate the container as in step 5. Because of the pickle mirror, enrollments made while on SQLite are already in the pickles.

If the mirror was disabled, enrollments made after cutover exist only in `attendance.db`. In that case, keep the rollback window short and re-enroll those people after rolling back.

## Later: retire the pickles

After about 30 days without problems:

1. Set `STORAGE_MIRROR_PICKLE=0` in `app.env` and recreate the container.
2. Archive the pickles encrypted and remove them from `data/`.
3. A later PR removes the pickle code path.

## Backups once on SQLite

Copy the live database consistently (safe while the service runs):

```bash
docker exec face-recognition-api python -c "import sqlite3; s=sqlite3.connect('/code/data/attendance.db'); d=sqlite3.connect('/code/data/attendance-backup.db'); s.backup(d); d.close()"
```

Move the backup off `data/`, encrypted, and test a restore monthly. Never copy `attendance.db` alone with `cp` while the service runs: the `-wal` file holds recent writes.
