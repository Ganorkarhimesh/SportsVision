import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference.predict import predict_video


# ---------------------------------------------------------
# Application Setup
# ---------------------------------------------------------

app = FastAPI(
    title="SportsVision API",
    description="Human Action Recognition in Sports Videos",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "SportsVision API",
        "status": "running"
    }


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": True
    }


# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file provided."
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mpg"
    }

    file_extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if file_extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV, or MKV."
            )
        )


    # -----------------------------------------------------
    # Create temporary input filename
    # -----------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_filename = (
        f"{unique_id}{file_extension}"
    )

    input_path = (
        UPLOAD_DIR / input_filename
    )


    # -----------------------------------------------------
    # Save uploaded video
    # -----------------------------------------------------

    try:

        contents = await file.read()

        with open(
            input_path,
            "wb"
        ) as video_file:

            video_file.write(contents)


        print(
            f"Received video: {file.filename}"
        )

        print(
            f"Saved to: {input_path}"
        )


        # -------------------------------------------------
        # Run SportsVision inference
        # -------------------------------------------------

        result = predict_video(
            str(input_path)
        )


        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result
        }


    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    finally:

        # -------------------------------------------------
        # Delete uploaded video
        # -------------------------------------------------

        if input_path.exists():

            os.remove(input_path)

            print(
                "Temporary video deleted."
            )