from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import time
import cv2
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy import ndimage
from scipy.stats import linregress
import math
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

def select_best_mask(masks, scores, box_prompt, image_shape):
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
        if intersection_ratio > 0.7 and area_ratio < 1.5:
            adjusted_score = original_score * (intersection_ratio + 0.5)
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

def calculate_elevation_anomalies(dsm_2d, mask, plane_params, threshold_factor=2.0):
    """
    Identify pixels that deviate significantly from the fitted plane
    
    Args:
        dsm_2d: 2D array of DSM elevation data
        mask: Binary mask of the roof segment
        plane_params: Parameters of the fitted plane (a, b, c) in z = ax + by + c
        threshold_factor: Multiplier for standard deviation threshold
        
    Returns:
        Binary mask of anomalous pixels
    """
    if plane_params is None or len(plane_params) < 3:
        print("Invalid plane parameters, returning empty anomaly mask")
        return np.zeros_like(mask, dtype=np.uint8)
    
    try:
        a, b, c = plane_params
        height, width = dsm_2d.shape
        mask_height, mask_width = mask.shape
        
        # Create a mask for anomalies
        anomaly_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Print dimensions for debugging
        print(f"DSM dimensions: {width}x{height}, Mask dimensions: {mask_width}x{mask_height}")
        
        # Calculate expected elevation for each point on the plane
        residuals = []
        residual_coords = []
        
        # Use the smaller of the dimensions to avoid index errors
        safe_height = min(height, mask_height)
        safe_width = min(width, mask_width)
        
        print(f"Using safe dimensions for analysis: {safe_width}x{safe_height}")
        
        for y in range(safe_height):
            for x in range(safe_width):
                if mask[y, x] > 0:
                    actual_elevation = dsm_2d[y, x]
                    expected_elevation = a*x + b*y + c
                    residual = actual_elevation - expected_elevation
                    
                    # Only include valid residuals (not NaN or infinity)
                    if np.isfinite(residual):
                        residuals.append(residual)
                        residual_coords.append((x, y))
        
        if not residuals:
            print("No valid residuals found, returning empty anomaly mask")
            return anomaly_mask
            
        # Calculate statistics of residuals
        residuals = np.array(residuals)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        print(f"Residual statistics: mean={mean_residual:.2f}, std={std_residual:.2f}")
        
        # Set threshold for anomalies
        # Positive threshold for elements that stick up from the roof
        threshold = mean_residual + threshold_factor * std_residual
        
        print(f"Anomaly threshold: {threshold:.2f}")
        
        # Mark pixels that exceed the threshold
        for i, (x, y) in enumerate(residual_coords):
            if residuals[i] > threshold:
                anomaly_mask[y, x] = 1
        
        # Apply morphological operations to clean up the anomaly mask if it has any anomalies
        anomaly_count = np.sum(anomaly_mask)
        print(f"Found {anomaly_count} anomalous pixels before morphology")
        
        if anomaly_count > 0:
            kernel = np.ones((3, 3), np.uint8)
            # Use try-except in case morphology operations fail
            try:
                # Opening (erosion followed by dilation) removes small isolated pixels
                anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
                # Closing (dilation followed by erosion) fills small holes
                anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
                print(f"Anomaly count after morphology: {np.sum(anomaly_mask)}")
            except Exception as morph_error:
                print(f"Error during morphological operations: {morph_error}")
                # Continue with the unprocessed mask if morphology fails
                pass
            
        return anomaly_mask
    
    except Exception as e:
        print(f"Error calculating elevation anomalies: {e}")
        traceback.print_exc()  # Print full stack trace for debugging
        return np.zeros_like(mask, dtype=np.uint8)

def polygon_to_mask(polygon, shape):
    """
    Convert polygon coordinates to a binary mask
    
    Args:
        polygon: List of points [{'x': x, 'y': y}, ...]
        shape: (height, width) tuple for the mask shape
        
    Returns:
        Binary mask as numpy array
    """
    try:
        mask = np.zeros(shape, dtype=np.uint8)
        
        if not polygon or len(polygon) < 3:
            return mask
            
        # Extract points
        points = np.array([[p['x'], p['y']] for p in polygon], dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [points], 1)
        
        return mask
    except Exception as e:
        print(f"Error converting polygon to mask: {e}")
        return np.zeros(shape, dtype=np.uint8)

def extract_dsm_data_from_mask(dsm_array, mask, dsm_width, dsm_height):
    """
    Extract DSM elevation data for pixels inside a mask
    """
    # Create a 2D view of the DSM data
    dsm_2d = np.array(dsm_array).reshape(dsm_height, dsm_width)
    
    # Check if mask dimensions match DSM dimensions, resize if necessary
    if mask.shape[0] != dsm_height or mask.shape[1] != dsm_width:
        # Convert boolean mask to uint8 before resizing (THIS IS THE FIX)
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_uint8, (dsm_width, dsm_height), interpolation=cv2.INTER_NEAREST)
        # Convert back to binary mask after resizing
        mask_resized = (mask_resized > 127).astype(np.uint8)
    else:
        mask_resized = mask.astype(np.uint8)  # Ensure consistent type
    
    # Extract coordinates and elevation values for pixels inside the mask
    coords = []
    elevations = []
    
    for y in range(dsm_height):
        for x in range(dsm_width):
            if mask_resized[y, x] > 0:
                coords.append((x, y))
                elevations.append(dsm_2d[y, x])
    
    return {
        'coordinates': coords,
        'elevations': elevations,
        'count': len(coords)
    }

def fit_plane_to_points(points, elevations):
    """
    Fit a 3D plane to points with elevation data
    
    Args:
        points: List of (x, y) coordinates
        elevations: List of z values corresponding to points
        
    Returns:
        Tuple of (plane_params, error) where plane_params is (a, b, c) in z = ax + by + c
    """
    if len(points) < 3:
        return None, float('inf')
    
    try:
        # Convert to numpy arrays
        points_array = np.array(points)
        elevations_array = np.array(elevations)
        
        # Filter out invalid values
        valid_idx = ~np.isnan(elevations_array)
        if np.sum(valid_idx) < 3:
            return None, float('inf')
            
        points_array = points_array[valid_idx]
        elevations_array = elevations_array[valid_idx]
        
        # Create A matrix for least squares fit
        A = np.column_stack((points_array, np.ones(len(points_array))))
        
        # Solve for plane coefficients
        plane_params, residuals, _, _ = np.linalg.lstsq(A, elevations_array, rcond=None)
        
        # Calculate error
        error = residuals[0] if len(residuals) > 0 else float('inf')
        
        return plane_params, error
    except Exception as e:
        print(f"Error fitting plane to points: {e}")
        return None, float('inf')

def calculate_plane_slope(plane_params):
    """
    Calculate slope and aspect from plane parameters
    
    Args:
        plane_params: Tuple of (a, b, c) in z = ax + by + c
        
    Returns:
        Dictionary with slope in degrees and aspect (azimuth) in degrees
    """
    if plane_params is None or len(plane_params) < 3:
        return {'slope_degrees': 0, 'aspect_degrees': 0}
    
    try:
        a, b, c = plane_params
        
        # Calculate slope (angle from horizontal)
        slope_radians = math.atan(math.sqrt(a*a + b*b))
        slope_degrees = math.degrees(slope_radians)
        
        # Calculate aspect (direction of slope)
        aspect_radians = math.atan2(-a, b)  # Negating a for proper orientation
        aspect_degrees = math.degrees(aspect_radians)
        
        # Convert to 0-360 range
        if aspect_degrees < 0:
            aspect_degrees += 360
            
        return {
            'slope_degrees': slope_degrees,
            'aspect_degrees': aspect_degrees
        }
    except Exception as e:
        print(f"Error calculating plane slope: {e}")
        return {'slope_degrees': 0, 'aspect_degrees': 0}

def find_anomaly_boxes(anomaly_mask):
    """
    Find bounding boxes around connected components in anomaly mask
    
    Args:
        anomaly_mask: Binary mask of anomalous pixels
        
    Returns:
        List of bounding boxes as dictionaries with min_x, min_y, max_x, max_y
    """
    try:
        # Ensure mask is proper type for OpenCV
        if anomaly_mask is None or anomaly_mask.size == 0:
            print("Anomaly mask is empty or None")
            return []
            
        # Convert boolean mask to uint8 if needed
        if anomaly_mask.dtype == bool:
            print("Converting boolean mask to uint8")
            anomaly_mask = anomaly_mask.astype(np.uint8)
        
        # Ensure we have a binary mask (0 or 1 values only)
        if anomaly_mask.max() > 1:
            anomaly_mask = (anomaly_mask > 0).astype(np.uint8)
        
        # Check if mask has any positive values
        if np.sum(anomaly_mask) == 0:
            print("Anomaly mask has no positive values")
            return []
            
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(anomaly_mask)
        
        print(f"Found {num_labels-1} connected components in anomaly mask")
        
        # Filter components by size
        min_size = 10  # Minimum anomaly size in pixels
        max_size = max(100, int(0.3 * np.sum(anomaly_mask)))  # Maximum 30% of roof area, at least 100px
        
        anomaly_boxes = []
        for i in range(1, num_labels):  # Skip background (0)
            size = stats[i, cv2.CC_STAT_AREA]
            
            if size < min_size or size > max_size:
                continue
                
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Add some margin
            margin_x = max(2, int(0.1 * w))
            margin_y = max(2, int(0.1 * h))
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(anomaly_mask.shape[1] - 1, x + w + margin_x)
            y2 = min(anomaly_mask.shape[0] - 1, y + h + margin_y)
            
            anomaly_boxes.append({
                'min_x': float(x1),
                'min_y': float(y1),
                'max_x': float(x2),
                'max_y': float(y2),
                'area': float(size)
            })
        
        print(f"Returning {len(anomaly_boxes)} filtered anomaly boxes")
        return anomaly_boxes
    except Exception as e:
        print(f"Error finding anomaly boxes: {e}")
        traceback.print_exc()  # Print full stack trace for debugging
        return []

@router.post("/predict")
async def predict_roof_segments(request: PredictionRequest):
    global sam_model
    
    # Print request summary for debugging
    print(f"Received request for building_id: {request.building_id}")
    print(f"Building box present: {request.building_box is not None}")
    print(f"Roof segments count: {len(request.roof_segments) if request.roof_segments else 0}")
    print(f"DSM data present: {request.dsm_data is not None}")
    
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
        
        # Prepare DSM data if available
        dsm_data = None
        dsm_2d = None
        if request.dsm_data is not None:
            try:
                # Extract DSM dimensions
                dsm_width = request.dsm_data.dimensions.get('width')
                dsm_height = request.dsm_data.dimensions.get('height')
                rgb_height, rgb_width = rgb_image.shape[:2]
                
                if dsm_width and dsm_height and request.dsm_data.elevationData:
                    # Create 2D array from flattened data
                    dsm_data = np.array(request.dsm_data.elevationData)
                    
                    # Verify dimensions match and reshape
                    if len(dsm_data) == dsm_width * dsm_height:
                        # Reshape to 2D array using the provided dimensions
                        print(f"Reshaping DSM data to {dsm_height}x{dsm_width}")
                        dsm_2d = dsm_data.reshape(dsm_height, dsm_width)
                        
                        # Verify dimensions match the RGB image
                        if dsm_width != rgb_width or dsm_height != rgb_height:
                            print(f"⚠️ Warning: DSM dimensions ({dsm_width}x{dsm_height}) don't match RGB dimensions ({rgb_width}x{rgb_height})")
                    else:
                        print(f"Error: DSM data length {len(dsm_data)} doesn't match expected dimensions {dsm_width}x{dsm_height}={dsm_width*dsm_height}")
            except Exception as e:
                print(f"Error processing DSM data, continuing without it: {e}")
                traceback.print_exc()
                dsm_data = None
                dsm_2d = None
        
        # Set the image for the model
        sam_model.set_image(rgb_image)
        
        # Initialize results collections
        all_roof_segments = []
        all_obstructions = []
        
        # Get building box area for significance calculations
        building_area = 0
        if request.building_box:
            building_area = (request.building_box.max_x - request.building_box.min_x) * \
                            (request.building_box.max_y - request.building_box.min_y)
        
        # STEP 1: Sort roof segments by area and process them
        if request.roof_segments and len(request.roof_segments) > 0:
            print(f"Using {len(request.roof_segments)} roof segments as prompts")
            
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
                is_group = segment.is_group if segment.is_group is not None else False
                member_count = segment.member_count if segment.member_count else 0
                member_ids = segment.member_ids if segment.member_ids else []
                
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
                
                # Select the best mask that represents a roof face (not negative space)
                mask, score = select_best_mask(masks, scores, box_prompt, rgb_image.shape[:2])
                
                if mask is not None:
                    # Convert mask to polygon
                    polygon = mask_to_polygon(mask)
                    
                    if polygon and len(polygon) >= 3:
                        # Simplify the polygon to use straight lines and fewer points
                        simplified_polygon = simplify_polygon(polygon, tolerance=2.0, max_points=8)
                        
                        # Calculate segment area (use the simplified polygon)
                        segment_area = calculate_area(simplified_polygon)
                        
                        # Add segment to results if it's large enough
                        if segment_area > 25:  # Minimum area threshold in pixels
                            segment_result = {
                                'id': segment_id,  # Use the original ID
                                'polygon': simplified_polygon,  # Use simplified polygon
                                'area': segment_area,
                                'confidence': float(score),
                                'has_obstruction': False,
                                'azimuth': segment_azimuth,
                                'pitch': segment_pitch,
                            }
                            
                            # Store mask as a separate attribute for later use
                            segment_result['_mask'] = mask
                            
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
                                'has_obstruction': False,
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
                            'has_obstruction': False,
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
                        'has_obstruction': False,
                        'azimuth': segment_azimuth,
                        'pitch': segment_pitch,
                        'error': 'No valid masks generated by model'
                    })
            
            # STEP 2: Now we use DSM data to detect obstructions within roof segments
            if dsm_2d is not None:
                print("Starting DSM-based obstruction detection")
                
                obstruction_id_counter = 0
                for segment in all_roof_segments:
                    # Skip segments without valid polygons or masks
                    if not segment.get('polygon') or len(segment.get('polygon', [])) < 3 or '_mask' not in segment:
                        continue
                    
                    segment_id = segment['id']
                    roof_mask = segment['_mask']
                    
                    # Step 2.1: Extract DSM data for this roof segment
                    segment_dsm_data = extract_dsm_data_from_mask(
                        dsm_data, roof_mask, dsm_2d.shape[1], dsm_2d.shape[0])
                    
                    if segment_dsm_data['count'] < 10:  # Need enough points to fit a plane
                        print(f"Not enough DSM data points for segment {segment_id}, skipping")
                        continue
                    
                    # Step 2.2: Fit a plane to the DSM data
                    plane_params, fit_error = fit_plane_to_points(
                        segment_dsm_data['coordinates'],
                        segment_dsm_data['elevations']
                    )
                    
                    if plane_params is None:
                        print(f"Failed to fit plane to DSM data for segment {segment_id}")
                        continue
                    
                    # Step 2.3: Calculate slope and aspect of the fitted plane
                    slope_info = calculate_plane_slope(plane_params)
                    
                    # Update segment with calculated slope and aspect
                    segment['calculated_pitch'] = slope_info['slope_degrees']
                    segment['calculated_azimuth'] = slope_info['aspect_degrees']
                    
                    # Step 2.4: Find elevation anomalies (potential obstructions)
                    anomaly_mask = calculate_elevation_anomalies(
                        dsm_2d, roof_mask, plane_params, threshold_factor=1.5)
                    
                    # Step 2.5: Find bounding boxes around anomalies
                    anomaly_boxes = find_anomaly_boxes(anomaly_mask)
                    
                    print(f"Detected {len(anomaly_boxes)} potential obstructions in segment {segment_id} using DSM data")
                    
                    # Step 2.6: Process each anomaly with SAM to get precise segmentation
                    for anomaly_box in anomaly_boxes:
                        obstruction_id_counter += 1
                        obstruction_id = f"obstruction_{obstruction_id_counter}"
                        
                        # Create box prompt for this obstruction
                        box_prompt = np.array([
                            anomaly_box['min_x'], 
                            anomaly_box['min_y'], 
                            anomaly_box['max_x'], 
                            anomaly_box['max_y']
                        ])
                        
                        try:
                            # Predict with the box prompt
                            masks, scores, _ = sam_model.predict(box=box_prompt)
                            
                            # Take the highest confidence mask
                            if len(masks) > 0:
                                best_idx = np.argmax(scores)
                                obstruction_mask = masks[best_idx]
                                score = scores[best_idx]
                                
                                # Make sure the obstruction mask is contained within the segment mask
                                obstruction_mask = np.logical_and(obstruction_mask, roof_mask).astype(np.uint8)
                                
                                if np.sum(obstruction_mask) == 0:
                                    continue  # Skip if no overlap with segment
                                
                                # Convert mask to polygon
                                polygon = mask_to_polygon(obstruction_mask)
                                
                                if polygon and len(polygon) >= 3:
                                    # Calculate area
                                    area = calculate_area(polygon)
                                    
                                    # Add to obstructions list if large enough
                                    if area > 10:  # Smaller threshold for obstructions
                                        all_obstructions.append({
                                            'id': obstruction_id,
                                            'polygon': polygon,
                                            'area': area,
                                            'confidence': float(score),
                                            'is_obstruction': True,
                                            'parent_segment': segment_id,
                                            'elevation_diff': float(anomaly_box.get('elevation_diff', 0.0))
                                        })
                        except Exception as e:
                            print(f"Error processing obstruction {obstruction_id}: {e}")
            else:
                print("No DSM data available for obstruction detection")
            
            # STEP 3: Mark roof segments with obstructions
            for segment in all_roof_segments:
                # Find obstructions belonging to this segment
                segment_obstructions = [o for o in all_obstructions if o.get('parent_segment') == segment.get('id')]
                
                if segment_obstructions:
                    segment['has_obstruction'] = True
                    segment['obstruction_count'] = len(segment_obstructions)
            
            # Resolve overlaps between segments
            all_roof_segments = resolve_overlaps_simple(all_roof_segments)
            
            # Remove masks from final output - we don't need to send them back
            for segment in all_roof_segments:
                if '_mask' in segment:
                    del segment['_mask']
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        return {
            "building_id": request.building_id,
            "roof_segments": all_roof_segments,
            "obstructions": all_obstructions,
            "status": "success",
            "processing_time_seconds": processing_time,
            "num_segments": len(all_roof_segments),
            "num_obstructions": len(all_obstructions),
            "used_roof_segments": request.roof_segments is not None and len(request.roof_segments) > 0,
            "used_building_box": request.building_box is not None,
            "used_dsm_data": dsm_data is not None
        }
    except Exception as e:
        # Log the full error details
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")