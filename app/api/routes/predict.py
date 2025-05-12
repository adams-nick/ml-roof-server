from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
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
    id: Optional[str] = None
    azimuth: Optional[float] = None
    pitch: Optional[float] = None
    is_group: Optional[bool] = False
    member_ids: Optional[List[str]] = None
    member_count: Optional[int] = None
    
    # Allow additional fields
    class Config:
        extra = "allow"

class PredictionRequest(BaseModel):
    building_id: str
    rgb_image: str  # Base64 encoded image
    building_box: Optional[BoundingBox] = None
    roof_segments: Optional[List[BoundingBox]] = []
    
    # Allow additional fields
    class Config:
        extra = "allow"

@router.post("/predict")
async def predict_roof_segments(request: PredictionRequest):
    global sam_model
    
    # Print request summary for debugging
    print(f"Received request for building_id: {request.building_id}")
    print(f"Building box present: {request.building_box is not None}")
    print(f"Roof segments count: {len(request.roof_segments) if request.roof_segments else 0}")
    
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
        
        # Initialize the segment collection
        all_roof_segments = []
        segment_id_counter = 1
        
        # Check if we have roof segments to use as prompts
        if request.roof_segments and len(request.roof_segments) > 0:
            print(f"Using {len(request.roof_segments)} roof segments as prompts")
            
            # Process each roof segment as a separate prompt
            for segment in request.roof_segments:
                # Create box prompt for this segment
                box_prompt = np.array([
                    segment.min_x, 
                    segment.min_y, 
                    segment.max_x, 
                    segment.max_y
                ])
                
                # Get segment metadata
                segment_id = segment.id if segment.id else f"segment_{segment_id_counter}"
                segment_azimuth = segment.azimuth
                segment_pitch = segment.pitch
                is_group = segment.is_group if segment.is_group is not None else False
                member_count = segment.member_count if segment.member_count else 0
                member_ids = segment.member_ids if segment.member_ids else []
                
                # Predict with this segment's bounding box
                masks, scores, _ = sam_model.predict(box=box_prompt)
                
                # Process each mask for this segment
                segment_results = []
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    # Convert mask to polygon
                    polygon = mask_to_polygon(mask)
                    
                    if polygon:
                        # Calculate segment area
                        segment_area = calculate_area(polygon)
                        
                        # Add segment to results if it's large enough
                        if segment_area > 25:  # Minimum area threshold in pixels
                            segment_result = {
                                'id': f"{segment_id}_roof_{i+1}",
                                'source_segment_id': segment_id,
                                'polygon': polygon,
                                'area': segment_area,
                                'confidence': float(score),
                                'has_obstruction': False,  # Will be updated in future pass
                                'azimuth': segment_azimuth,
                                'pitch': segment_pitch,
                            }
                            
                            # Add group information if applicable
                            if is_group:
                                segment_result['is_from_group'] = True
                                segment_result['group_member_count'] = member_count
                                segment_result['group_member_ids'] = member_ids
                            
                            segment_results.append(segment_result)
                
                # Add results from this segment to the overall collection
                all_roof_segments.extend(segment_results)
                segment_id_counter += 1
        
        # If no roof segments were provided or no valid segments were found,
        # fall back to using the building box as a prompt
        if (not request.roof_segments or len(all_roof_segments) == 0) and request.building_box:
            print("Using building box as fallback prompt")
            
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
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Convert mask to polygon
                polygon = mask_to_polygon(mask)
                
                if polygon:
                    # Calculate segment area
                    segment_area = calculate_area(polygon)
                    
                    # Add segment to results if it's large enough
                    if segment_area > 25:  # Minimum area threshold in pixels
                        all_roof_segments.append({
                            'id': f"roof_{i+1}",
                            'polygon': polygon,
                            'area': segment_area,
                            'confidence': float(score),
                            'has_obstruction': False,  # Will be updated in future pass
                            'from_building_box': True  # Flag that this came from building box
                        })
        
        # Make sure we have at least some segments
        if len(all_roof_segments) == 0:
            print("Warning: No valid roof segments were generated")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        return {
            "building_id": request.building_id,
            "roof_segments": all_roof_segments,
            "status": "success",
            "processing_time_seconds": processing_time,
            "num_segments": len(all_roof_segments),
            "used_roof_segments": request.roof_segments is not None and len(request.roof_segments) > 0,
            "used_building_box": request.building_box is not None and (request.roof_segments is None or len(all_roof_segments) == 0)
        }
    except Exception as e:
        # Log the full error details
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")