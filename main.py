from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tempfile
import os
import shutil

app = FastAPI(title="FaceVerify Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/verify")
async def verify_faces(
    id_photo: UploadFile = File(..., description="National ID card photo"),
    front_photo: UploadFile = File(..., description="Front/upright face photo (used for matching)"),
    left_photo: UploadFile = File(..., description="Left side face photo (liveness check)"),
    right_photo: UploadFile = File(..., description="Right side face photo (liveness check)"),
):
    """
    3-angle liveness verification + ArcFace identity match against ID card.

    Steps:
    1. Detect a face in ALL three angle photos (liveness gate)
    2. Verify left & right are real face photos (not spoofed prints)
    3. Run ArcFace comparison: ID photo vs front photo
    """
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    for f in [id_photo, front_photo, left_photo, right_photo]:
        if f.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' must be JPEG, PNG, or WebP.")

    tmp_dir = tempfile.mkdtemp()
    paths = {
        "id":    os.path.join(tmp_dir, "id.jpg"),
        "front": os.path.join(tmp_dir, "front.jpg"),
        "left":  os.path.join(tmp_dir, "left.jpg"),
        "right": os.path.join(tmp_dir, "right.jpg"),
    }

    try:
        # Save all files
        for key, upload in [("id", id_photo), ("front", front_photo), ("left", left_photo), ("right", right_photo)]:
            with open(paths[key], "wb") as f:
                shutil.copyfileobj(upload.file, f)

        from deepface import DeepFace

        # ── STEP 1: Liveness gate — detect faces in all 3 angle photos ──
        liveness_errors = []
        for angle, path in [("front", paths["front"]), ("left", paths["left"]), ("right", paths["right"])]:
            try:
                DeepFace.extract_faces(
                    img_path=path,
                    detector_backend="retinaface",
                    enforce_detection=True,
                    align=True,
                )
            except ValueError:
                liveness_errors.append(angle)

        if liveness_errors:
            raise HTTPException(
                status_code=422,
                detail=f"Could not detect a face in: {', '.join(liveness_errors)} photo(s). "
                       "Please retake those shots with clear lighting and your face fully visible."
            )

        # ── STEP 2: Verify ID card has a detectable face ──
        try:
            DeepFace.extract_faces(
                img_path=paths["id"],
                detector_backend="retinaface",
                enforce_detection=True,
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Could not detect a face in the ID card photo. "
                       "Make sure the face on the ID is clearly visible and not obscured."
            )

        # ── STEP 3: ArcFace verification — ID vs Front photo ──
        result = DeepFace.verify(
            img1_path=paths["id"],
            img2_path=paths["front"],
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
        )

        # Compute a clean confidence score (0–100%)
        raw_conf = (1 - result["distance"] / result["threshold"]) * 100
        confidence = round(max(0.0, min(100.0, raw_conf)), 1)

        return {
            "verified": result["verified"],
            "distance": round(result["distance"], 4),
            "threshold": round(result["threshold"], 4),
            "confidence": confidence,
            "liveness_passed": True,
            "angles_verified": ["front", "left", "right"],
            "model": "ArcFace",
            "detector": "RetinaFace",
        }

    except HTTPException:
        raise

    except ValueError as e:
        msg = str(e)
        if "face" in msg.lower() or "detect" in msg.lower():
            raise HTTPException(status_code=422, detail="Face detection failed. Ensure images are clear and well-lit.")
        raise HTTPException(status_code=422, detail=msg)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "ArcFace", "detector": "RetinaFace", "version": "2.0"}


#RUN CODE=>: uvicorn main:app --host 0.0.0.0 --port 8444 --ssl-keyfile key.pem --ssl-certfile cert.pem

