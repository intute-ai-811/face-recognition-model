"""
Dump detections, enrollment results and recognition scores for a set of images
to JSON, using a throwaway data directory (never touches data/).

Run it under two runtimes (or two code versions) and diff the outputs to prove
an upgrade or refactor did not change recognition behaviour:

    python scripts/pipeline_snapshot.py --enroll a.jpg b.jpg --probe c.jpg --out before.json
    python scripts/pipeline_snapshot.py --enroll a.jpg b.jpg --probe c.jpg --out after.json
    python scripts/pipeline_snapshot.py --compare before.json after.json

Output contains scores and boxes only, no images or embeddings, but it still
describes real people: keep it out of git.
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def snapshot(enroll: list[str], probe: list[str], person_id: str) -> dict:
    from app import config

    tmp = Path(tempfile.mkdtemp(prefix="pipeline_snapshot_"))
    config.DATA_DIR = tmp
    config.EMBED_DIR = tmp / "embeddings"
    config.EMBED_INDEX_FILE = tmp / "embeddings" / "index.pkl"
    config.PEOPLE_FILE = tmp / "people.pkl"
    config.FACES_DIR = tmp / "faces"

    import cv2
    from app.core.recognize import Pipeline

    p = Pipeline()
    out = {"detect": {}, "enroll": {}, "recognize": {}}
    images = {Path(f).name: cv2.imread(f) for f in enroll + probe}
    for name, img in images.items():
        b, k = p.det.detect(img)
        out["detect"][name] = {"boxes": b.tolist(), "kps": k.tolist()}
    for f in enroll:
        out["enroll"][Path(f).name] = p.enroll_image(images[Path(f).name], person_id=person_id, source=Path(f).name)
    for name, img in images.items():
        out["recognize"][name] = [
            {"person_id": r["prediction"]["person_id"], "similarity": r["prediction"]["similarity"], "bbox": r["bbox"]}
            for r in p.recognize_image(img)
        ]
    return out


def _max_abs_diff(a, b) -> float:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"keys differ: {sorted(a)} vs {sorted(b)}"
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        assert len(a) == len(b), f"length differs: {len(a)} vs {len(b)}"
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    assert a == b, f"{a!r} != {b!r}"
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--person-id", default="1")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed abs difference in any number")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads(Path(p).read_text()) for p in args.compare)
        diff = _max_abs_diff(a, b)
        ok = diff <= args.tol
        print(f"structure identical; max abs numeric difference {diff:.3e} -> {'OK' if ok else 'DIFFERENT'}")
        sys.exit(0 if ok else 1)

    logging.disable(logging.CRITICAL)
    Path(args.out).write_text(json.dumps(snapshot(args.enroll, args.probe, args.person_id), indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
