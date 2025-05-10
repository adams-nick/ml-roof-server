from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import time

from app.models.sam_model import SamModel
from app.utils.image_utils import decode_base64_image, mask_to_polygon, calculate_area

# Create a global instance of the SAM model
# This will be loaded only once when the server starts
sam_model = None

router = APIRouter()

class BoundingBox(BaseModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

class PredictionRequest(BaseModel):
    building_id: str
    rgb_image: str  # Base64 encoded image
    building_box: BoundingBox

@router.post("/predict")
async def predict_roof_segments(request: PredictionRequest):
    global sam_model
    
    # Initialize model if not already done
    if sam_model is None:
        try:
            sam_model = SamModel()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize SAM model: {str(e)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Decode the base64 image
        try:
            rgb_image = decode_base64_image(request.rgb_image)
            print(f"Decoded image shape: {rgb_image.shape}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
        
        # Set the image for the model
        sam_model.set_image(rgb_image)
        
        # Use the building bounding box as the prompt
        box_prompt = np.array([
            request.building_box.min_x, 
            request.building_box.min_y, 
            request.building_box.max_x, 
            request.building_box.max_y
        ])
        
        # Predict roof segments
        masks, scores, _ = sam_model.predict(box=box_prompt)
        
        # Process each mask
        roof_segments = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Convert mask to polygon
            polygon = mask_to_polygon(mask)
            
            if polygon:
                # Calculate segment area
                segment_area = calculate_area(polygon)
                
                # Add segment to results if it's large enough
                if segment_area > 25:  # Minimum area threshold in pixels
                    roof_segments.append({
                        'id': f"roof_{i+1}",
                        'polygon': polygon,
                        'area': segment_area,
                        'confidence': float(score),
                        'has_obstruction': False  # Will be updated in future pass
                    })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        return {
            "building_id": request.building_id,
            "roof_segments": roof_segments,
            "status": "success",
            "processing_time_seconds": processing_time,
            "num_segments": len(roof_segments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
