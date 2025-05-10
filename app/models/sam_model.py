import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SamModel:
    def __init__(self, model_type="vit_h", checkpoint_path="models/sam_vit_h_4b8939.pth"):
        """
        Initialize the SAM model
        
        Args:
            model_type: The type of SAM model to use ('vit_h', 'vit_b', 'vit_l')
            checkpoint_path: Path to the model checkpoint
        """
        print(f"Initializing SAM model ({model_type})...")
        
        # Determine device (use CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the SAM model
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            self.model_loaded = True
            print("SAM model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading SAM model: {str(e)}")
            raise RuntimeError(f"Failed to load SAM model: {str(e)}")
    
    def set_image(self, image):
        """
        Set the image for SAM predictor
        
        Args:
            image: RGB image as a numpy array with shape (H, W, 3)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            self.predictor.set_image(image)
            return True
        except Exception as e:
            print(f"Error setting image: {str(e)}")
            return False
    
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None):
        """
        Run SAM prediction with the given prompts
        
        Args:
            point_coords: Optional point coordinates as numpy array of shape (N, 2)
            point_labels: Optional point labels as numpy array of shape (N,)
            box: Optional box as numpy array of shape (4,)
            mask_input: Optional mask input
            
        Returns:
            masks, scores, logits
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure inputs are in the right format
        if point_coords is not None and not isinstance(point_coords, np.ndarray):
            point_coords = np.array(point_coords)
        
        if point_labels is not None and not isinstance(point_labels, np.ndarray):
            point_labels = np.array(point_labels)
        
        if box is not None and not isinstance(box, np.ndarray):
            box = np.array(box)
        
        # Run prediction
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=True
            )
            return masks, scores, logits
        except Exception as e:
            print(f"Error running prediction: {str(e)}")
            raise RuntimeError(f"Failed to run prediction: {str(e)}")
