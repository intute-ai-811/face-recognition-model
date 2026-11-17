"""
Accuracy evaluation of the recognition pipeline on a consented dataset.

Dataset layout (keep it OUT of git; it is biometric data):

    DATASET/
      enroll/<person_id>/*.jpg     enrollment images per person
      probe/<person_id>/*.jpg      later images of the same enrolled people (genuine attempts)
      impostor/**/*.jpg            people who are NOT enrolled (unknown-face attempts)

The pipeline is run exactly as in production (same detector, enrollment rules,
max-cosine matching, first prediction like /mark_attendance) against a
throwaway store; data/ is never touched.

Reported (open-set identification, as an attendance kiosk works):
  FPIR  false positive identification rate: impostor attempts accepted as someone
  FNIR  false negative identification rate: genuine attempts not accepted as the right person
  misID genuine attempts accepted as the WRONG person (included in FNIR, and a false accept)
  EER   threshold where FPIR ~= FNIR
  TAR @ FPIR 1% / 0.1%: share of genuine attempts correctly accepted at the threshold
        that lets through at most that share of impostors

    python benchmarks/eval_accuracy.py DATASET --json report.json
    python benchmarks/eval_accuracy.py DATASET --threshold 0.50

The JSON report contains aggregate numbers, score histograms, the model SHA-256,
the git commit and a dataset fingerprint - no images, embeddings or person IDs
(unless --details is given, in which case keep the report private too).
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Attempt:
    kind: str                 # "genuine" | "impostor"
    true_id: str | None       # None for impostors
    top_id: str | None        # best-matching enrolled id, None if no face / empty DB
    score: float              # similarity of top_id (0.0 if none)
    face_found: bool


# ── metrics (pure functions; unit-tested) ────────────────────────────────────

def rates_at(attempts: list[Attempt], t: float) -> dict:
    gen = [a for a in attempts if a.kind == "genuine"]
    imp = [a for a in attempts if a.kind == "impostor"]
    correct = sum(1 for a in gen if a.face_found and a.score >= t and a.top_id == a.true_id)
    misid = sum(1 for a in gen if a.face_found and a.score >= t and a.top_id != a.true_id)
    imp_accept = sum(1 for a in imp if a.face_found and a.score >= t)
    return {
        "threshold": round(t, 4),
        "genuine": len(gen),
        "impostor": len(imp),
        "TAR": correct / len(gen) if gen else None,
        "FNIR": 1 - correct / len(gen) if gen else None,
        "misID": misid / len(gen) if gen else None,
        "FPIR": imp_accept / len(imp) if imp else None,
    }


def eer(attempts: list[Attempt]) -> dict | None:
    if not any(a.kind == "impostor" for a in attempts) or not any(a.kind == "genuine" for a in attempts):
        return None
    best = None
    for t in np.linspace(0, 1, 1001):
        r = rates_at(attempts, float(t))
        gap = abs(r["FPIR"] - r["FNIR"])
        if best is None or gap < best[0]:
            best = (gap, r)
    r = best[1]
    return {"threshold": r["threshold"], "EER": (r["FPIR"] + r["FNIR"]) / 2}


def tar_at_fpir(attempts: list[Attempt], target: float) -> dict | None:
    imp = sorted((a.score for a in attempts if a.kind == "impostor" and a.face_found), reverse=True)
    n_imp = sum(1 for a in attempts if a.kind == "impostor")
    if not n_imp:
        return None
    allowed = int(np.floor(target * n_imp))          # impostors we may let through
    # Smallest threshold that accepts at most `allowed` impostors.
    t = (imp[allowed] + 1e-6) if allowed < len(imp) else 0.0
    r = rates_at(attempts, t)
    return {"target_FPIR": target, "threshold": r["threshold"], "TAR": r["TAR"], "FPIR": r["FPIR"],
            "note": None if n_imp * target >= 1 else f"only {n_imp} impostor attempts; FPIR {target} not resolvable"}


def histogram(scores: list[float], bins: int = 20) -> dict:
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    return {"edges": [round(float(e), 3) for e in edges], "counts": counts.tolist()}


# ── running the pipeline ─────────────────────────────────────────────────────

def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def dataset_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in images_in(root):
        h.update(str(p.relative_to(root)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def run(root: Path, details: bool) -> tuple[list[Attempt], dict]:
    from app import config

    tmp = Path(tempfile.mkdtemp(prefix="eval_"))
    config.DATA_DIR = tmp
    config.EMBED_DIR = tmp / "embeddings"
    config.EMBED_INDEX_FILE = tmp / "embeddings" / "index.pkl"
    config.PEOPLE_FILE = tmp / "people.pkl"
    config.FACES_DIR = tmp / "faces"

    import cv2
    from app.core.recognize import Pipeline

    pipe = Pipeline()
    enroll_stats = {"people": 0, "images": 0, "embeddings": 0, "people_below_min": 0}
    for person_dir in sorted(p for p in (root / "enroll").iterdir() if p.is_dir()):
        n_emb = 0
        for img_path in images_in(person_dir):
            img = cv2.imread(str(img_path))
            enroll_stats["images"] += 1
            if img is None:
                continue
            n_emb += pipe.enroll_image(img, person_id=person_dir.name, source=img_path.name)["embeddings_added"]
        enroll_stats["people"] += 1
        enroll_stats["embeddings"] += n_emb
        if n_emb < config.ENROLL_MIN_GOOD_EMBEDDINGS:
            enroll_stats["people_below_min"] += 1

    attempts: list[Attempt] = []
    per_image = []

    def attempt(img_path: Path, kind: str, true_id: str | None):
        img = cv2.imread(str(img_path))
        preds = pipe.recognize_image(img) if img is not None else []
        if not preds:
            a = Attempt(kind, true_id, None, 0.0, False)
        else:
            # Same choice as /mark_attendance: the first prediction. Use the raw best
            # candidate (top_k[0]) so scores below the threshold are still measured.
            top = preds[0]["top_k"][0] if preds[0]["top_k"] else None
            a = Attempt(kind, true_id, top["person_id"] if top else None,
                        float(top["score"]) if top else 0.0, True)
        attempts.append(a)
        if details:
            per_image.append({"file": str(img_path.relative_to(root)), **a.__dict__})

    probe_dir = root / "probe"
    if probe_dir.exists():
        for person_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
            for img_path in images_in(person_dir):
                attempt(img_path, "genuine", person_dir.name)
    imp_dir = root / "impostor"
    if imp_dir.exists():
        for img_path in images_in(imp_dir):
            attempt(img_path, "impostor", None)

    meta = {"enrollment": enroll_stats}
    if details:
        meta["per_image"] = per_image
    return attempts, meta


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold (default: config.SIMILARITY_THRESHOLD)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--details", action="store_true", help="include per-image results (contains person IDs)")
    args = ap.parse_args()

    if not (args.dataset / "enroll").is_dir():
        raise SystemExit(f"{args.dataset}/enroll/ not found - see the layout in --help")

    logging.disable(logging.CRITICAL)
    from app import config

    t = config.SIMILARITY_THRESHOLD if args.threshold is None else args.threshold
    attempts, meta = run(args.dataset, args.details)
    gen_scores = [a.score for a in attempts if a.kind == "genuine" and a.face_found and a.top_id == a.true_id]
    imp_scores = [a.score for a in attempts if a.kind == "impostor" and a.face_found]

    report = {
        "git_sha": git_sha(),
        "model_sha256": hashlib.sha256(Path(config.MOBILEFACENET_MODEL).read_bytes()).hexdigest(),
        "dataset_fingerprint": dataset_fingerprint(args.dataset),
        **meta,
        "no_face": {
            "genuine": sum(1 for a in attempts if a.kind == "genuine" and not a.face_found),
            "impostor": sum(1 for a in attempts if a.kind == "impostor" and not a.face_found),
        },
        "at_operating_threshold": rates_at(attempts, t),
        "EER": eer(attempts),
        "TAR_at_FPIR_1pct": tar_at_fpir(attempts, 0.01),
        "TAR_at_FPIR_0.1pct": tar_at_fpir(attempts, 0.001),
        "genuine_score_hist": histogram(gen_scores),
        "impostor_score_hist": histogram(imp_scores),
        "genuine_score_range": [min(gen_scores), max(gen_scores)] if gen_scores else None,
        "impostor_score_max": max(imp_scores) if imp_scores else None,
    }
    op = report["at_operating_threshold"]
    print(f"threshold {op['threshold']}: genuine {op['genuine']}, impostor {op['impostor']}")
    print(f"  TAR {op['TAR']}  FNIR {op['FNIR']}  misID {op['misID']}  FPIR {op['FPIR']}")
    print(f"  EER {report['EER']}")
    print(f"  genuine score range {report['genuine_score_range']}, max impostor score {report['impostor_score_max']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
