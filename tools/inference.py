import numpy as np
import torch
import os
from tools.torch_stuff import get_segformer3d_plus
from tools.postprocessing import postprocess_segmentation

# Initialize device and model once at module level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and checkpoint
model = get_segformer3d_plus()
checkpoint_path = "checkpoints/best_segformer3d_model.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpnt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpnt["model_state_dict"])
model = model.to(device)
model.eval()

print("Model loaded and ready for inference")


def segment_image(image):
    """
    Perform segmentation on preprocessed image.
    
    Args:
        image: Preprocessed image tensor (expected shape: [1, 4, H, W, D] or [4, H, W, D])
    
    Returns:
        numpy array: Post-processed segmentation mask
    """
    global model, device
    
    with torch.no_grad():
        # Ensure image is on correct device and has batch dimension
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        image = image.to(device)
        
        # Add batch dimension if not present
        if image.dim() == 4:  # [4, H, W, D]
            image = image.unsqueeze(0)  # [1, 4, H, W, D]
        
        print(f"Input image shape: {image.shape}")
        
        # Model inference
        output = model(image)
        
        # Handle different output formats
        if isinstance(output, tuple):
            logits = output[0]
        elif isinstance(output, dict):
            logits = output.get('logits', output.get('out', output))
        else:
            logits = output
        
        # Get prediction
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        print(f"Raw prediction shape: {prediction.shape}")
        print(f"Unique values in prediction: {np.unique(prediction)}")
        
        # Post-process prediction
        final_prediction = postprocess_segmentation(prediction)
        
        print(f"Final prediction shape: {final_prediction.shape}")
        print(f"Unique values in final prediction: {np.unique(final_prediction)}")

        return final_prediction
