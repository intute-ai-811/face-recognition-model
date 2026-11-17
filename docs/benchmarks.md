# Benchmarks

Baseline for plan PR 5 (see `docs/MODERNIZATION_PLAN.md`). Every later performance or model change is compared against these numbers, measured on the same machine.

> **Status (2026-11-17):** the speed, memory and load baselines below are from a **developer machine**, not the VPS. The **accuracy** baseline is still pending, because it needs consented photos from the actual kiosks (see [Accuracy](#accuracy-pending-kiosk-dataset)).

## Environment

| | Value |
| --- | --- |
| Machine | Developer workstation: Intel Core Ultra 9 285K, 22 threads, WSL2 |
| Runtime | Python 3.13.18, onnxruntime 1.32.0, numpy 2.5.5 (PR 4 stack) |
| Models | SCRFD-2.5G (`041f73f4…`), MobileFaceNet `w600k_mbf` (`9cc6e4a7…`) |
| Code | branch `pr5-benchmarks` (PR 1–4 applied); recognition behaviour identical to production |

The VPS has fewer and slower cores, so expect higher absolute latencies there. Run the same commands on the VPS to get the numbers that matter.

## Pipeline stage latency

`benchmarks/bench_pipeline.py --images rhythm1.jpeg test.jpg --repeat 20`, one face per image, one request at a time.

| Stage | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| Detect (SCRFD at 640×640) | 23.4 – 27.4 | 34 – 40 |
| Align (5-point) | < 0.1 | < 0.1 |
| Embed (MobileFaceNet) | 10.7 – 12.0 | 15 – 16 |
| **Full `recognize_image`** | **41.7 – 44.0** | **63.5 – 64.7** |

The process uses about 260 MB RSS after the models load (0.1 s model load).

## Matching at scale

Synthetic random embeddings, 10 templates per person, one query, the current `search_cosine`.

| People | Templates | Search p50 (ms) | Search p95 (ms) | Index memory (MB) |
| --- | --- | --- | --- | --- |
| 100 | 1,000 | 0.14 | 0.19 | 0.3 |
| 1,000 | 10,000 | 1.44 | 3.49 | 18 |
| 10,000 | 100,000 | 17.2 | 17.8 | 218 |
| 100,000 | 1,000,000 | 178 | 180 | 2,291 |

**Reading:** matching is negligible at factory scale (≤ 1,000 people), so it does not justify FAISS or a vector database. Cost grows linearly with template count because of the per-template Python loop (finding F16). Memory is about twice the raw vector size because embeddings are held both in the records and in the normalised matrix. Both are cheap to fix (vectorised per-person max; one copy) if the site count ever grows; neither is needed now.

## Shift-change load test

`benchmarks/locustfile.py`: 50 kiosk users all start at once, each posting an image to `/mark_attendance` and then waiting 1–3 s. The server is one uvicorn worker. ERP is `benchmarks/mock_erp.py`, so no real attendance was marked. Two probe images alternate; the enrolled person matches, so every request goes through the ERP path.

| Requests | Failures | Throughput | p50 | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| 1,048 | 0 | 23.8 req/s | 120 ms | 250 ms | 630 ms | 693 ms |

**Reading:** a 50-person burst is served with no errors and every request under 0.7 s on this machine. The tail (p99 630 ms against a 120 ms median) comes from requests queueing behind each other on the CPU. Thread tuning (finding F19, PR 6) should be judged against this table on the VPS.

Raw results: `benchmarks/results/baseline-dev-2026-11-17.json`, `benchmarks/results/shift-change-dev-2026-11-17.csv`.

## PR 6: insightface removed, warm-up, thread setting

Detection and embedding now run on in-house ONNX Runtime wrappers; `insightface` and about 30 packages are gone. Outputs are unchanged:

| Check | Result |
| --- | --- |
| `tests/test_detector_equivalence.py` (vs insightface 0.7.3) | boxes, scores, landmarks within 1e-3 px; embeddings within 1e-5 |
| Golden embeddings | pass without insightface installed |
| `scripts/pipeline_snapshot.py` vs the original production stack (Python 3.11 + insightface) | identical within 1.2e-7; `test.jpg` 0.575 |

**Warm-up** (`MODEL_WARMUP=1`, default): the first request after startup took 33 ms instead of 64 ms.

**ONNX Runtime threads under a 50-user burst** (same load test as above, 30 s, dev machine):

| `ORT_INTRA_OP_THREADS` | Requests | Failures | Throughput | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| unset (ORT default = all cores) | 702 | 0 | 24.2 req/s | 120 ms | 480 ms | 730 ms |
| 4 | 735 | 0 | 25.4 req/s | 40 ms | 320 ms | 390 ms |
| 2 | 712 | 0 | 24.6 req/s | 60 ms | 310 ms | 380 ms |

**Reading:** with many requests in flight, letting each inference use every core makes them fight over the CPU. Capping threads per inference cut the median about 3x and the p99 about 2x here, with the same throughput. The default is left unchanged because the VPS CPU is different. To adopt it, run the load test on the VPS with a few values and then set the best one in `app.env` (e.g. `ORT_INTRA_OP_THREADS=2` on a 2–4 vCPU VPS). No code change or rebuild is needed.

## Accuracy (pending kiosk dataset)

`benchmarks/eval_accuracy.py` is ready but has not been run on real kiosk data. Smoke test on the repository's sample photos (1 person, 2 enrollment images, 2 probes, no impostors): TAR 1.0 at 0.50, genuine scores 0.53–0.81. That is far too small to mean anything; it only shows the script works end to end.

When kiosk photos become available:

1. Collect with consent: at least 50 enrolled people with 5+ probe photos each (different shifts and lighting), plus at least 30 people who are **not** enrolled.
2. Lay them out outside the repository:

    ```text
    DATASET/
      enroll/<person_id>/*.jpg
      probe/<person_id>/*.jpg
      impostor/**/*.jpg
    ```

3. Run `uv run python benchmarks/eval_accuracy.py DATASET --json accuracy-baseline.json`.
4. Commit only the JSON report (aggregates, histograms, model hash, dataset fingerprint). Never commit the images.

Until then, the model switch (plan PR 15) and the enrollment-rule changes (plan PR 13) stay blocked. Nothing else in the plan depends on this baseline.

## How to run

All commands run from a checkout with `uv sync` done. None of them touch `data/`.

```bash
# Stage latency on real images (needs models/)
uv run python benchmarks/bench_pipeline.py --images test.jpg rhythm1.jpeg --json stages.json

# Matching at scale (no models needed)
uv run python benchmarks/bench_pipeline.py --matching-only --people 100 1000 10000 100000

# Shift-change load test - local or staging only, never production
uv run python benchmarks/mock_erp.py --port 9999 &
ERP_ATTENDANCE_URL=http://127.0.0.1:9999/attendance \
  uv run uvicorn app.api.server:app --host 127.0.0.1 --port 6911 &
LOAD_TEST_IMAGES=test.jpg:rhythm1.jpeg uvx locust -f benchmarks/locustfile.py \
  --host http://127.0.0.1:6911 --headless -u 50 -r 50 -t 45s --csv shift

# Accuracy (consented dataset, kept out of git)
uv run python benchmarks/eval_accuracy.py DATASET --json accuracy.json

# Before/after equivalence check for any runtime or refactor change
uv run python scripts/pipeline_snapshot.py --enroll a.jpg b.jpg --probe c.jpg --out before.json
uv run python scripts/pipeline_snapshot.py --compare before.json after.json
```

Note: the load-test server above uses the checkout's own `data/` directory. Run it from a scratch copy of `app/` (as done for this baseline) if that directory holds anything you care about.
