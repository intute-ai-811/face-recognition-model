#app/core/recognize.py
import logging
import numpy as np
from app.core.detector import FaceDetector
from app.core.align import align_5pts
from app.core.embedder import FaceEmbedder
from app.core.storage import LocalStore
from app.logging_config import timed, get_logger
from app import config

log = get_logger(__name__)


class Pipeline:
    def __init__(self):
        self.det = FaceDetector()
        self.emb = FaceEmbedder()
        self.db = LocalStore()

    def enroll_image(self, img, person_id, source):
        log.info("enroll_image_in", extra={"person_id": person_id, "source": source})

        with timed(log, "enroll_detect"):
            bboxes, kps = self.det.detect(img)

        aligned = []
        for i in range(len(bboxes)):
            a = align_5pts(img, kps[i])
            if a is None:
                log.warning("enroll_align_none", extra={"person_id": person_id, "idx": i})
                continue
            aligned.append(a)

        log.info("enroll_align", extra={"aligned": len(aligned)})

        with timed(log, "enroll_embed", faces=len(aligned)):
            embs = self.emb.embed(aligned)

        added = self.db.add_embeddings(person_id, embs, source)
        log.info(
            "enroll_image_out",
            extra={"person_id": person_id, "faces": len(aligned), "embeddings": added},
        )
        return {"faces": len(aligned), "embeddings_added": added}

    def enroll_video(
        self,
        video_bytes,
        person_id,
        source,
        sample_every=config.VIDEO_SAMPLE_EVERY,
        min_face_size_px=config.ENROLL_MIN_FACE_PX,
        max_frames_considered=config.VIDEO_MAX_FRAMES,
        max_templates=config.ENROLL_MAX_TEMPLATES,
    ):
        log.info("enroll_video_in", extra={"person_id": person_id, "source": source})

        from app.core.video import iter_video_frames_from_bytes, pick_diverse_indices
        from app.core.quality import face_quality_scores, passes_quality

        frames = iter_video_frames_from_bytes(
            video_bytes,
            sample_every=sample_every,
            max_frames=max_frames_considered,
        )

        total_detected = 0
        aligned_faces = []
        qinfo = []

        for fr in frames:
            bboxes, kps = self.det.detect(fr)
            if bboxes.shape[0] == 0:
                continue
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            idx = int(np.argmax(areas))
            w = bboxes[idx][2] - bboxes[idx][0]
            h = bboxes[idx][3] - bboxes[idx][1]
            if min(w, h) < min_face_size_px:
                continue
            a = align_5pts(fr, kps[idx])
            if a is None:
                log.warning("enroll_video_align_none", extra={"person_id": person_id})
                continue
            q = face_quality_scores(a)
            if not passes_quality(q):
                continue
            aligned_faces.append(a)
            qinfo.append(q)
            total_detected += 1

        log.info(
            "video_selected_faces",
            extra={"detected": total_detected, "kept_quality": len(aligned_faces)},
        )

        embs = self.emb.embed(aligned_faces)
        ranked = sorted(
            range(len(aligned_faces)),
            key=lambda i: (qinfo[i]["sharpness"], -abs(qinfo[i]["brightness"] - 128)),
            reverse=True,
        )
        embs_ranked = embs[ranked]
        idx_keep = pick_diverse_indices(
            embs_ranked,
            max_keep=max_templates,
            dedupe_cosine=config.DIVERSITY_COSINE_TOL,
        )
        final_embs = embs_ranked[idx_keep]
        final_faces = [aligned_faces[ranked[i]] for i in idx_keep]

        for a in final_faces:
            self.db.save_aligned(person_id, a)

        added = self.db.add_embeddings(person_id, final_embs, source)
        log.info(
            "enroll_video_out",
            extra={
                "person_id": person_id,
                "frames": len(frames),
                "faces_kept": len(final_faces),
                "embeddings": int(added),
            },
        )
        return {
            "frames_total": len(frames),
            "frames_detected": total_detected,
            "faces_kept": len(final_faces),
            "embeddings_added": int(added),
        }

    def recognize_image(self, img):
        log.info("recognize_in")

        with timed(log, "recognize_detect"):
            bboxes, kps = self.det.detect(img)

        # Guard against None returns from align_5pts.
        # A None aligned face would produce a zero-vector embedding
        # giving cosine similarity ~0 against everything → always "unknown".
        aligned = []
        valid_bboxes = []
        for i, k in enumerate(kps):
            a = align_5pts(img, k)
            if a is None:
                log.warning("recognize_align_none", extra={"face_idx": i})
                continue
            aligned.append(a)
            valid_bboxes.append(bboxes[i])

        log.info("recognize_align", extra={"aligned": len(aligned)})

        if not aligned:
            return []

        with timed(log, "recognize_embed", faces=len(aligned)):
            embs = self.emb.embed(aligned)

        sims = self.db.search_cosine(embs, top_k=config.TOP_K)

        out = []
        for i in range(len(valid_bboxes)):
            best = sims[i][0] if sims[i] else {"person_id": "unknown", "score": 0.0}
            label = best["person_id"] if best["score"] >= config.SIMILARITY_THRESHOLD else "unknown"
            out.append(
                {
                    "bbox": valid_bboxes[i][:4].tolist(),
                    "score": float(valid_bboxes[i][4]),
                    "prediction": {"person_id": label, "similarity": best["score"]},
                    "top_k": sims[i],
                }
            )

        log.info("recognize_out", extra={"faces": len(out)})
        return out