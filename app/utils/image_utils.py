import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import json
from shapely.geometry import Polygon

def decode_base64_image(base64_string):
    """Decode a base64 image to a numpy array"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    image = np.array(Image.open(BytesIO(image_data)))
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

def mask_to_polygon(mask, simplify_tolerance=1.0):
    """Convert binary mask to polygon coordinates"""
    # Ensure mask is binary
    mask = mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert to polygon points
    points = []
    for point in largest_contour.reshape(-1, 2):
        points.append({"x": float(point[0]), "y": float(point[1])})
    
    # Simplify if requested (using Shapely)
    if simplify_tolerance > 0 and len(points) > 3:
        try:
            # Convert to Shapely polygon
            coords = [(p["x"], p["y"]) for p in points]
            poly = Polygon(coords)
            
            # Simplify
            poly_simple = poly.simplify(simplify_tolerance)
            
            # Convert back to points format
            points = [{"x": float(x), "y": float(y)} 
                     for x, y in poly_simple.exterior.coords]
        except Exception as e:
            print(f"Error simplifying polygon: {e}")
    
    return points

def calculate_area(polygon):
    """Calculate area of a polygon in square pixels"""
    if not polygon or len(polygon) < 3:
        return 0
    
    # Convert to numpy array
    points = np.array([[p["x"], p["y"]] for p in polygon])
    
    # Calculate area using shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return float(area)
