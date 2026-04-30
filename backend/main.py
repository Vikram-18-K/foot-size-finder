from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import traceback
import logging

from cv_pipeline import process_image, fast_validate_image, CVError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Foot Size Measurement API")

# Setup CORS to allow the frontend to communicate with this backend map
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MeasurementRequest(BaseModel):
    image_base64: str
    foot_side: str  # "left" or "right"

class MeasurementResponse(BaseModel):
    success: bool
    length_cm: float
    width_cm: float
    shoe_size_uk: float
    shoe_size_us: float
    message: str

class ValidationResponse(BaseModel):
    a4_detected: bool
    foot_detected: bool
    is_blurry: bool
    tilt_ok: bool
    valid: bool
    confidence: float
    message: str

@app.post("/api/validate", response_model=ValidationResponse)
async def validate_foot_scan(request: MeasurementRequest):
    try:
        # 1. Decode the Base64 image
        encoded_data = request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64
        image_data = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # 2. Decode into cv2 image format
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return ValidationResponse(
                a4_detected=False, foot_detected=False, is_blurry=True, 
                tilt_ok=False, valid=False, confidence=0.0, 
                message="Failed to decode image."
            )
            
        # 3. Fast Validation
        results = fast_validate_image(img, request.foot_side)
        return ValidationResponse(**results)
        
    except Exception as e:
        logger.error(f"Validation Error: {str(e)}")
        return ValidationResponse(
            a4_detected=False, foot_detected=False, is_blurry=True, 
            tilt_ok=False, valid=False, confidence=0.0, 
            message="Internal validation error."
        )

@app.post("/api/measure", response_model=MeasurementResponse)
async def measure_foot(request: MeasurementRequest):
    try:
        # 1. Decode the Base64 image
        encoded_data = request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64
        image_data = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # 2. Decode into cv2 image format
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
            
        # 3. Pass through the robust CV Pipeline
        results = process_image(img, request.foot_side)
        
        return MeasurementResponse(
            success=True,
            length_cm=results["length_cm"],
            width_cm=results["width_cm"],
            shoe_size_uk=results["shoe_size_uk"],
            shoe_size_us=results["shoe_size_us"],
            message="Measurement successful"
        )
        
    except CVError as cv_err:
        logger.warning(f"CV Validation failed: {str(cv_err)}")
        return MeasurementResponse(
            success=False,
            length_cm=0.0,
            width_cm=0.0,
            shoe_size_uk=0.0,
            shoe_size_us=0.0,
            message=str(cv_err)
        )
    except ValueError as val_e:
        logger.error(f"Value Error: {str(val_e)}")
        return MeasurementResponse(
            success=False,
            length_cm=0.0,
            width_cm=0.0,
            shoe_size_uk=0.0,
            shoe_size_us=0.0,
            message=str(val_e)
        )
    except Exception as e:
        logger.error(f"Server Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Run the app locally if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
