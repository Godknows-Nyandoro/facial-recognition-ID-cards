# FaceVerify — ID Authentication System

Face verification app using ArcFace + RetinaFace via DeepFace, served with FastAPI.

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the server**
```bash
uvicorn main:app --reload --port 8000
```

**3. Open in browser**
Visit: http://localhost:8000

## How It Works

1. Upload a National ID card photo (left panel)
2. Open the webcam and capture three live photos (facing right, facing left, and upright) to ensure a real person is present and avoid verifying an ID without an actual face
3. Click **Verify Identity**
4. The backend runs ArcFace face verification and returns:
   - ✅ Match / ❌ No Match verdict
   - Distance score (lower = more similar)
   - Threshold used
   - Confidence percentage

## API Endpoint

`POST /verify`

**Form data:**
- `id_photo` — image file (JPEG/PNG/WebP)
- `live_photo1` — image file (right-facing)
- `live_photo2` — image file (left-facing)
- `live_photo3` — image file (upright)

**Response:**
```json
{
  "verified": true,
  "distance": 0.2341,
  "threshold": 0.68,
  "similarity_percent": 65.6,
  "model": "ArcFace",
  "detector": "RetinaFace"
}
```

## Project Structure
```
face_verify/
├── main.py              # FastAPI backend
├── requirements.txt     # Python dependencies
├── README.md
└── static/
    └── index.html       # Frontend UI
```

## Notes

- First run downloads ArcFace + RetinaFace model weights (~500MB)
- Ensure faces are clearly visible and well-lit
- Users should use the webcam for live captures to prevent spoofing with static photos
- For ID cards: crop the photo area if the card has a lot of text/borders
- Models run on CPU by default; GPU speeds it up significantly
