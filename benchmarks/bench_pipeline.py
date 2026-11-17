"""
Latency benchmark for the recognition pipeline stages and the matcher.

Two parts:
  1. Stages (detect / align / embed / full recognize) on real images you pass in.
     Needs the model files. Prints timings only; no images, embeddings or
     identities are written.
  2. Matching at scale (100 / 1k / 10k / 100k people) with synthetic random
     embeddings: search latency and memory of the in-memory index.
     No model files needed.

Examples:
    python benchmarks/bench_pipeline.py --images rhythm1.jpeg test.jpg
    python benchmarks/bench_pipeline.py --matching-only --people 100 1000 10000 100000
    python benchmarks/bench_pipeline.py --images test.jpg --json results.json

Run on the target machine (the VPS) for numbers that matter; a dev laptop is
only useful for before/after comparisons on the same machine.
"""
import argparse
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "p50_ms": round(pct(ms, 50), 2),
        "p95_ms": round(pct(ms, 95), 2),
        "p99_ms": round(pct(ms, 99), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
    }


def timeit(fn, repeat: int, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def isolate_data_dir() -> None:
    """Point storage at a temp dir so benchmarks never touch data/."""
    from app import config

    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    config.DATA_DIR = tmp
    config.EMBED_DIR = tmp / "embeddings"
    config.EMBED_INDEX_FILE = tmp / "embeddings" / "index.pkl"
    config.PEOPLE_FILE = tmp / "people.pkl"
    config.FACES_DIR = tmp / "faces"


def bench_stages(images: list[str], repeat: int) -> dict:
    import cv2
    from app.core.align import align_5pts
    from app.core.recognize import Pipeline

    rss0 = rss_mb()
    t0 = time.perf_counter()
    pipe = Pipeline()
    load_s = time.perf_counter() - t0
    # One enrolled template so recognize() exercises matching too.
    first = cv2.imread(images[0])
    pipe.enroll_image(first, person_id="1", source="bench")

    results = {"model_load_s": round(load_s, 2), "rss_after_load_mb": round(rss_mb(), 1),
               "rss_model_delta_mb": round(rss_mb() - rss0, 1), "images": {}}
    for path in images:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        bboxes, kps = pipe.det.detect(img)
        r = {"shape": list(img.shape), "faces": int(len(bboxes))}
        r["detect"] = summarize(timeit(lambda: pipe.det.detect(img), repeat))
        if len(bboxes):
            r["align"] = summarize(timeit(lambda: align_5pts(img, kps[0]), repeat))
            aligned = align_5pts(img, kps[0])
            r["embed"] = summarize(timeit(lambda: pipe.emb.embed([aligned]), repeat))
        r["recognize_total"] = summarize(timeit(lambda: pipe.recognize_image(img), repeat))
        results["images"][Path(path).name] = r
    return results


def bench_matching(people_counts: list[int], templates: int, queries: int) -> dict:
    from app import config
    from app.core.storage import EmbRecord, LocalStore

    rng = np.random.default_rng(0)
    out = {}
    for n in people_counts:
        isolate_data_dir()
        store = LocalStore()
        total = n * templates
        embs = rng.standard_normal((total, config.EMB_DIM)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        # Build in memory directly (persisting 1M records per add would dominate).
        rss_before = rss_mb()
        store._records = [EmbRecord(str(i // templates), embs[i], 0.0, "bench") for i in range(total)]
        store._rebuild_cache()
        rss_after = rss_mb()
        q = rng.standard_normal((1, config.EMB_DIM)).astype(np.float32)
        reps = max(3, min(queries, int(2000 / max(1, total / 1000))))
        ms = timeit(lambda: store.search_cosine(q, top_k=config.TOP_K), reps, warmup=1)
        out[str(n)] = {"templates": total, "search": summarize(ms),
                       "index_rss_delta_mb": round(rss_after - rss_before, 1)}
        print(f"  {n:>7} people x {templates} templates: p50 {out[str(n)]['search']['p50_ms']} ms, "
              f"p95 {out[str(n)]['search']['p95_ms']} ms, index ~{out[str(n)]['index_rss_delta_mb']} MB", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--matching-only", action="store_true")
    ap.add_argument("--people", nargs="*", type=int, default=[100, 1000, 10000, 100000])
    ap.add_argument("--templates", type=int, default=10, help="templates per person for the matching bench")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    isolate_data_dir()
    import onnxruntime

    report = {
        "env": {
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
        }
    }
    if not args.matching_only and args.images:
        print("Stage latency (ms):", flush=True)
        report["stages"] = bench_stages(args.images, args.repeat)
        for name, r in report["stages"]["images"].items():
            parts = ", ".join(f"{k} p50 {v['p50_ms']} / p95 {v['p95_ms']}" for k, v in r.items() if isinstance(v, dict))
            print(f"  {name} ({r['faces']} face): {parts}")
        print(f"  model load {report['stages']['model_load_s']} s, RSS {report['stages']['rss_after_load_mb']} MB")
    if args.people:
        print("Matching latency:", flush=True)
        report["matching"] = bench_matching(args.people, args.templates, args.queries)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
