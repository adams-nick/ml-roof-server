from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import time
import cv2
from shapely.geometry import Polygon
import traceback

from app.models.sam_model import SamModel
from app.utils.image_utils import decode_base64_image, mask_to_polygon, calculate_area

# Create a global instance of the SAM model
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
    orientation: Optional[str] = None
    suitability: Optional[float] = None
    is_group: Optional[bool] = False
    member_ids: Optional[List[str]] = None
    member_count: Optional[int] = None
    
    # Allow additional fields
    class Config:
        extra = "allow"

class DsmData(BaseModel):
    elevationData: List[float]
    dimensions: Dict[str, int]
    dataRange: Optional[Dict[str, float]] = None
    
    class Config:
        extra = "allow"

class PredictionRequest(BaseModel):
    building_id: str
    rgb_image: str  # Base64 encoded image
    building_box: Optional[BoundingBox] = None
    roof_segments: Optional[List[BoundingBox]] = []
    dsm_data: Optional[DsmData] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    # Allow additional fields
    class Config:
        extra = "allow"

def select_best_mask(masks, scores, box_prompt, image_shape, segment_properties=None):
    """
    Select the best mask that represents a roof face rather than negative space.
    Uses heuristics like intersection with box, area ratios, etc.
    """
    if masks is None or len(masks) == 0:
        return None, 0.0
    
    best_mask = None
    best_score = -1
    best_idx = -1
    
    # Create a binary mask from the box_prompt
    box_mask = np.zeros(image_shape, dtype=np.uint8)
    x1, y1, x2, y2 = box_prompt.astype(int)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)  # Fill the rectangle
    box_area = (x2 - x1) * (y2 - y1)
    
    for i, (mask, original_score) in enumerate(zip(masks, scores)):
        # Calculate how much of the mask is inside the box
        intersection = np.logical_and(mask, box_mask).sum()
        mask_area = mask.sum()
        
        # If mask is empty, skip
        if mask_area == 0:
            continue
            
        # Calculate various metrics
        intersection_ratio = intersection / mask_area  # How much of mask is in the box
        area_ratio = mask_area / box_area  # Size compared to the box
        
        # Skip masks that are too large compared to the box (likely negative space)
        if area_ratio > 2.0:
            continue
            
        # Prefer masks that have high overlap with the box and reasonable size
        adjusted_score = original_score * (intersection_ratio + 0.5)
        
        # If we have known roof properties, consider them for scoring
        if segment_properties:
            # Boost score based on suitability if available
            if segment_properties.get('suitability') is not None:
                adjusted_score *= (1.0 + segment_properties.get('suitability', 0.5))
        
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_mask = mask
            best_idx = i
    
    # If no good mask found with the above criteria, fallback to highest score mask
    if best_mask is None and len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
    
    return best_mask, float(best_score)

def simplify_polygon(polygon, tolerance=2.0, max_points=8):
    """
    Simplify a polygon to use straight lines and fewer points.
    
    Args:
        polygon: List of points [{'x': x, 'y': y}, ...]
        tolerance: Simplification tolerance - higher values create simpler polygons
        max_points: Target maximum number of points in the polygon
    
    Returns:
        Simplified polygon as list of points [{'x': x, 'y': y}, ...]
    """
    if not polygon or len(polygon) < 3:
        return polygon
    
    try:
        # Convert to numpy array
        points = np.array([[p['x'], p['y']] for p in polygon])
        
        # Convert to Shapely polygon
        # Close the ring if it's not already closed
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        
        shapely_poly = Polygon(points)
        
        # If polygon is not valid, try to fix it
        if not shapely_poly.is_valid:
            shapely_poly = shapely_poly.buffer(0)
            
            # If still invalid, return original
            if not shapely_poly.is_valid:
                return polygon
        
        # Try different tolerance values until we get a polygon with <= max_points
        current_tolerance = tolerance
        simplified = None
        
        # Using the built-in simplify method from shapely
        while True:
            simplified = shapely_poly.simplify(current_tolerance, preserve_topology=True)
            
            # Check if simplification resulted in a MultiPolygon
            if simplified.geom_type == 'MultiPolygon':
                # Take the largest polygon from the MultiPolygon
                simplified = max(simplified.geoms, key=lambda g: g.area)
            
            # Check number of points in the simplified polygon
            num_points = len(list(simplified.exterior.coords)) - 1  # -1 because first/last points are the same
            
            if num_points <= max_points or current_tolerance > 30:
                break
                
            current_tolerance *= 1.5
        
        # Get the exterior points (skipping the last point which duplicates the first)
        coords = list(simplified.exterior.coords)[:-1]
        
        # Convert back to the required format
        return [{'x': float(x), 'y': float(y)} for x, y in coords]
    
    except Exception as e:
        print(f"Error simplifying polygon: {e}")
        return polygon  # Return original if simplification fails

def resolve_overlaps_simple(segments):
    """
    Simple approach to resolve overlaps between polygon segments.
    Higher confidence segments take precedence over lower confidence ones.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        List of segments with overlaps resolved
    """
    if not segments or len(segments) < 2:
        return segments
    
    try:
        # Convert segments to Shapely polygons for easier processing
        shapely_segments = []
        for segment in segments:
            if not segment.get('polygon') or len(segment.get('polygon', [])) < 3:
                # Skip segments with no valid polygons
                shapely_segments.append(segment)
                continue
            
            try:
                # Convert to Shapely polygon
                coords = [(p['x'], p['y']) for p in segment['polygon']]
                poly = Polygon(coords)
                
                if poly.is_valid and not poly.is_empty:
                    # Store the segment with its Shapely polygon
                    segment_copy = segment.copy()
                    segment_copy['shapely_polygon'] = poly
                    shapely_segments.append(segment_copy)
                else:
                    # Invalid polygon, keep original segment
                    shapely_segments.append(segment)
            except Exception as e:
                print(f"Error creating Shapely polygon for segment {segment.get('id')}: {e}")
                shapely_segments.append(segment)
        
        # Sort segments by confidence (highest first)
        shapely_segments.sort(key=lambda s: s.get('confidence', 0), reverse=True)
        
        # Process segments in order of confidence
        result = []
        for i, segment in enumerate(shapely_segments):
            if 'shapely_polygon' not in segment:
                # No valid polygon, just add to result
                result.append(segment)
                continue
            
            # Check for overlaps with higher confidence segments
            current_poly = segment['shapely_polygon']
            original_poly = current_poly
            
            # Subtract all higher confidence polygons
            for j in range(i):
                if 'shapely_polygon' not in shapely_segments[j]:
                    continue
                
                higher_poly = shapely_segments[j]['shapely_polygon']
                if current_poly.intersects(higher_poly):
                    # Subtract the higher confidence polygon
                    try:
                        current_poly = current_poly.difference(higher_poly)
                    except Exception as e:
                        print(f"Error subtracting polygons: {e}")
                        # If subtraction fails, keep original polygon
                        current_poly = original_poly
                        break
            
            # If the resulting polygon is valid, convert back to the format we need
            if current_poly.is_valid and not current_poly.is_empty:
                # Handle both Polygon and MultiPolygon cases
                new_segment = segment.copy()
                del new_segment['shapely_polygon']
                
                if current_poly.geom_type == 'Polygon':
                    # Single polygon
                    coords = list(current_poly.exterior.coords)
                    # Remove last point if it duplicates the first
                    if coords[0] == coords[-1]:
                        coords = coords[:-1]
                    
                    new_segment['polygon'] = [{'x': float(x), 'y': float(y)} for x, y in coords]
                    new_segment['area'] = float(current_poly.area)
                    result.append(new_segment)
                else:
                    # MultiPolygon - take the largest part
                    try:
                        largest = max(current_poly.geoms, key=lambda p: p.area)
                        coords = list(largest.exterior.coords)
                        # Remove last point if it duplicates the first
                        if coords[0] == coords[-1]:
                            coords = coords[:-1]
                        
                        new_segment['polygon'] = [{'x': float(x), 'y': float(y)} for x, y in coords]
                        new_segment['area'] = float(largest.area)
                        result.append(new_segment)
                    except Exception as e:
                        print(f"Error handling MultiPolygon: {e}")
                        # If handling MultiPolygon fails, use original
                        new_segment['polygon'] = segment['polygon']
                        result.append(new_segment)
            else:
                # If the resulting polygon is invalid, keep the original segment's polygon
                new_segment = segment.copy()
                del new_segment['shapely_polygon']
                result.append(new_segment)
        
        return result
    
    except Exception as e:
        print(f"Error in overlap resolution: {e}")
        # Return original segments if anything goes wrong
        return [s for s in segments if 'shapely_polygon' not in s] + [
            {**s, 'polygon': s['polygon']} for s in segments if 'shapely_polygon' in s
        ]

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
        
        # Initialize results collections
        all_roof_segments = []
        
        # Get building box area for significance calculations
        building_area = 0
        if request.building_box:
            building_area = (request.building_box.max_x - request.building_box.min_x) * \
                            (request.building_box.max_y - request.building_box.min_y)
        
        # Process roof segments from the request
        if request.roof_segments and len(request.roof_segments) > 0:
            print(f"Processing {len(request.roof_segments)} roof segments")
            
            # Calculate area for each roof segment
            roof_segments_with_area = []
            for segment in request.roof_segments:
                segment_id = segment.id if segment.id else f"segment_{id(segment)}"
                segment_area = (segment.max_x - segment.min_x) * (segment.max_y - segment.min_y)
                significance = segment_area / building_area if building_area > 0 else 1.0
                
                roof_segments_with_area.append({
                    'segment': segment,
                    'area': segment_area,
                    'significance': significance,
                    'id': segment_id
                })
            
            # Sort by area (largest first)
            roof_segments_with_area.sort(key=lambda s: s['area'], reverse=True)
            
            # Process each roof segment in order of size
            for seg_info in roof_segments_with_area:
                segment = seg_info['segment']
                segment_id = seg_info['id']
                
                # Skip if significance is too low (less than 1% of building area)
                if seg_info['significance'] < 0.01 and building_area > 0:
                    print(f"Skipping segment {segment_id} due to low significance ({seg_info['significance']:.2f})")
                    continue
                
                # Create box prompt for this segment
                box_prompt = np.array([
                    segment.min_x, 
                    segment.min_y, 
                    segment.max_x, 
                    segment.max_y
                ])
                
                # Get segment metadata
                segment_azimuth = segment.azimuth
                segment_pitch = segment.pitch
                segment_orientation = segment.orientation if hasattr(segment, 'orientation') else None
                segment_suitability = segment.suitability if hasattr(segment, 'suitability') else None
                is_group = segment.is_group if segment.is_group is not None else False
                member_count = segment.member_count if segment.member_count else 0
                member_ids = segment.member_ids if segment.member_ids else []
                
                # Create properties dictionary for enhanced processing
                segment_properties = {
                    'pitch': segment_pitch,
                    'azimuth': segment_azimuth,
                    'orientation': segment_orientation,
                    'suitability': segment_suitability
                }
                
                # Add a center point prompt to help guide the model
                center_x = (segment.min_x + segment.max_x) / 2
                center_y = (segment.min_y + segment.max_y) / 2
                point_prompt = np.array([[center_x, center_y]])
                point_labels = np.array([1])  # 1 indicates foreground
                
                try:
                    # Try first with box and point prompt for better guidance
                    masks, scores, _ = sam_model.predict(
                        box=box_prompt,
                        point_coords=point_prompt,
                        point_labels=point_labels
                    )
                except Exception as e:
                    print(f"Error with combined prompt: {e}")
                    # Fallback to just box prompt
                    masks, scores, _ = sam_model.predict(box=box_prompt)
                
                # Select the best mask that represents a roof face
                mask, score = select_best_mask(masks, scores, box_prompt, rgb_image.shape[:2], segment_properties)
                
                if mask is not None:
                    # Convert mask to polygon
                    polygon = mask_to_polygon(mask)
                    
                    if polygon and len(polygon) >= 3:
                        # Simplify the polygon to use straight lines and fewer points
                        #simplified_polygon = simplify_polygon(polygon, tolerance=2.0, max_points=8)
                        simplified_polygon = polygon
                        # Calculate segment area (use the simplified polygon)
                        segment_area = calculate_area(simplified_polygon)
                        
                        # Add segment to results if it's large enough
                        if segment_area > 25:  # Minimum area threshold in pixels
                            segment_result = {
                                'id': segment_id,  # Use the original ID
                                'polygon': simplified_polygon,  # Use simplified polygon
                                'area': segment_area,
                                'confidence': float(score),
                                'azimuth': segment_azimuth,
                                'pitch': segment_pitch,
                            }
                            
                            # Add new properties if available
                            if segment_orientation is not None:
                                segment_result['orientation'] = segment_orientation
                            if segment_suitability is not None:
                                segment_result['suitability'] = segment_suitability
                            
                            # Add group information if applicable
                            if is_group:
                                segment_result['is_from_group'] = True
                                segment_result['group_member_count'] = member_count
                                segment_result['group_member_ids'] = member_ids
                            
                            all_roof_segments.append(segment_result)
                        else:
                            print(f"Polygon for segment {segment_id} too small (area: {segment_area})")
                            # Add a placeholder with the original ID
                            all_roof_segments.append({
                                'id': segment_id,
                                'polygon': [],  # Empty polygon
                                'area': 0,
                                'confidence': 0.0,
                                'azimuth': segment_azimuth,
                                'pitch': segment_pitch,
                                'error': 'Generated polygon too small'
                            })
                    else:
                        print(f"Unable to generate valid polygon for segment {segment_id}")
                        all_roof_segments.append({
                            'id': segment_id,
                            'polygon': [],  # Empty polygon
                            'area': 0,
                            'confidence': 0.0,
                            'azimuth': segment_azimuth,
                            'pitch': segment_pitch,
                            'error': 'Unable to generate valid polygon'
                        })
                else:
                    print(f"No valid masks generated for segment {segment_id}")
                    all_roof_segments.append({
                        'id': segment_id,
                        'polygon': [],  # Empty polygon
                        'area': 0,
                        'confidence': 0.0,
                        'azimuth': segment_azimuth,
                        'pitch': segment_pitch,
                        'error': 'No valid masks generated by model'
                    })
            
            # Resolve overlaps between segments
            all_roof_segments = resolve_overlaps_simple(all_roof_segments)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        return {
            "building_id": request.building_id,
            "roof_segments": all_roof_segments,
            "obstructions": [],  # Empty obstructions array
            "status": "success",
            "processing_time_seconds": processing_time,
            "num_segments": len(all_roof_segments),
            "num_obstructions": 0,
            "used_roof_segments": request.roof_segments is not None and len(request.roof_segments) > 0,
            "used_building_box": request.building_box is not None
        }
    except Exception as e:
        # Log the full error details
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")