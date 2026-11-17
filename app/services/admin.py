# app/services/admin.py
"""Admin operations on the enrollment store. Behaviour unchanged from the old server.py helpers."""
from __future__ import annotations

import pickle

from fastapi import HTTPException

from app import config


def people_dump(pipe) -> dict:
    """Embedding count per person, as returned by GET /admin/people."""
    try:
        if hasattr(pipe, "db") and hasattr(pipe.db, "people"):
            data = pipe.db.people()
            return dict(data) if isinstance(data, dict) else {"people": data}
    except Exception:
        pass
    # Legacy fallback kept for compatibility: reads the operator's own people.pkl.
    if not config.PEOPLE_FILE.exists():
        return {}
    with open(config.PEOPLE_FILE, "rb") as f:
        obj = pickle.load(f)
    return dict(obj) if isinstance(obj, dict) else {"people": obj}


def remove_person(pipe, person_id: str) -> dict:
    db = getattr(pipe, "db", None)
    for meth in ("remove_person", "delete_person", "drop_person"):
        if db is not None and hasattr(db, meth):
            return {"removed_via": f"PIPE.db.{meth}", "result": getattr(db, meth)(person_id)}
    raise HTTPException(status_code=500, detail="No deletion method found on storage layer.")
