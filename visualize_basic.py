import cv2
import numpy as np
import json
import sys
import matplotlib.pyplot as plt

def visualize_segments(image_path, json_path):
    """Visualize roof segments from a JSON file on the image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load the JSON data - read as string first
    with open(json_path, 'r') as f:
        json_content = f.read()
    
    # Try to find and parse the JSON part
    try:
        # Look for the start of the JSON object
        json_start = json_content.find('{')
        if json_start == -1:
            print("Error: No JSON object found in the file")
            return
        
        # Extract JSON part and parse it
        json_part = json_content[json_start:]
        result = json.loads(json_part)
        
        # Verify the structure
        if 'roof_segments' not in result:
            print("Error: No roof_segments found in the JSON data")
            print("JSON keys:", result.keys())
            return
        
        print(f"Found {len(result['roof_segments'])} roof segments in the JSON data")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("First 100 chars of content:", json_content[:100])
        return
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Show the image
    plt.imshow(img)
    
    # Colors for different segments
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
    
    # Draw each segment
    for i, segment in enumerate(result['roof_segments']):
        color = colors[i % len(colors)]
        
        # Extract polygon points
        if 'polygon' not in segment:
            print(f"Warning: Segment {i} has no polygon data")
            continue
            
        polygon = np.array([(p['x'], p['y']) for p in segment['polygon']])
        
        # Draw filled polygon with transparency
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3)
        
        # Draw outline
        plt.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=2, 
                label=f"Roof {i+1} (conf: {segment['confidence']:.2f})")
    
    # Add title and legend
    plt.title(f"Roof Segmentation Results ({len(result['roof_segments'])} segments)")
    plt.legend(loc='upper right')
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_basic.py <image_path> <json_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    json_path = sys.argv[2]
    
    visualize_segments(image_path, json_path)
