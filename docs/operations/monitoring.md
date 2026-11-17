# Monitoring

The API exposes Prometheus metrics, a readiness check and optional JSON logs. No metric label, log field or readiness response ever contains a face image, an embedding, a name or a person ID in a label. Logs contain person IDs, as before.

## Endpoints

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `GET /healthz` | Liveness, used by the Docker healthcheck and the deploy workflow | Response unchanged |
| `GET /readyz` | Readiness: 200 when the enrollment store answers, else 503 | `{"status": "ready", "checks": {"store": true, "embeddings": N}}` |
| `GET /metrics` | Prometheus text format | Set `METRICS_TOKEN` in `app.env` to require `Authorization: Bearer <token>`. Strongly recommended while port 6900 is reachable from outside (until plan PR 9 puts a proxy in front) |

## Metrics

| Metric | Labels | Meaning |
| --- | --- | --- |
| `face_api_http_requests_total` | method, route (template), status | Requests. Unknown paths are counted as `route="unmatched"` |
| `face_api_http_request_seconds` | route | Request latency histogram |
| `face_api_attendance_outcomes_total` | action (`check_in`/`check_out`), outcome | `marked`, `logged_out`, `already`, `no_match`, `no_face`, `mapping_error`, `invalid_upload`, `erp_error`, `erp_exception`, `error` |
| `face_api_pipeline_stage_seconds` | stage (`detect`/`embed`/`recognize`) | Inference time |
| `face_api_erp_request_seconds` | action | ERP call latency |
| `face_api_erp_failures_total` | action, kind (`exception`/`status`) | ERP calls that failed (after any retry) |
| `face_api_erp_retries_total` | action, reason (`connect`/`status_5xx`) | ERP calls retried after a transient failure (see below) |
| `face_api_enrollment_outcomes_total` | outcome | `enrolled`, `insufficient`, `no_valid_images`, `rejected` |
| `face_api_enrollment_images_total` | result | Per-image enrollment results |
| `face_api_store_embeddings`, `face_api_store_people` | none | Enrolled templates and people for the active model |

## Prometheus setup on the VPS

Add a scrape job. The target depends on how Prometheus runs: from the host it's `localhost:6900`; from a container on the same Docker network it's `face-recognition-api:6900`.

```yaml
scrape_configs:
  - job_name: face-api
    metrics_path: /metrics
    scrape_interval: 30s
    authorization:
      credentials: <METRICS_TOKEN>   # omit if METRICS_TOKEN is not set
    static_configs:
      - targets: ["localhost:6900"]
rule_files:
  - /etc/prometheus/face-api-alerts.yml   # copy of deploy/monitoring/alerts.yml
```

Alert rules (`deploy/monitoring/alerts.yml`):

| Alert | Fires when |
| --- | --- |
| FaceApiDown | scrape fails for 2 min |
| FaceApiNoEnrollmentsLoaded | 0 embeddings loaded for 5 min (data volume missing or the wrong backend) |
| FaceApiHighErrorRate | more than 5% 5xx over 10 min |
| FaceApiErpFailures | more than 3 ERP failures in 10 min |
| FaceApiSlowRecognition | attendance p95 over 2 s for 10 min |
| FaceApiMostlyNoFace | more than half of kiosk photos have no face over 30 min (camera problem) |

## JSON logs

Set `LOG_FORMAT=json` in `app.env` for one JSON object per line, which suits log shippers (Loki, Elastic). The default stays the existing `key=value` format.

## ERP retry

When the ERP is unreachable (connection refused/reset, DNS) or answers 502/503/504, the
service retries once after 0.5 s before giving the kiosk the usual 502. Timeouts are
not retried, so the kiosk never waits for two 8 s timeouts. A retry is safe because the
ERP answers a duplicate with `already_checked_in` / `already_checked_out`; in that case
the kiosk sees "Already checked in at <time>" instead of "Attendance marked successfully."

| Setting | Default | Meaning |
| --- | --- | --- |
| `ERP_RETRIES` | `1` | Extra attempts; `0` restores the previous behaviour |
| `ERP_RETRY_BACKOFF_S` | `0.5` | Wait before attempt *n* is `n × backoff` |

A steady rise in `face_api_erp_retries_total` with few `face_api_erp_failures_total`
means the ERP is flaky but the retry is absorbing it.
