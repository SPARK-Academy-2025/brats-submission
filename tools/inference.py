import numpy as np
import nibabel as nib
import torch
import os


from tools.torch_stuff import get_segformer3d_plus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_segformer3d_plus()
# 2. Load checkpoint
checkpnt = torch.load("checkpoints/best_segformer3d_model.pth", map_location=device)
prebuild_model = model.load_state_dict(checkpnt["model_state_dict"])


def segment_image(image, model=prebuild_model, device=device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        
        # Model inference
        output = model(image)
        logits = output[0] if isinstance(output, tuple) else output
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Post-process prediction
        final_prediction = postprocess_segmentation(prediction)

        return final_prediction

    print(f"All predictions saved to: {output_dir}")
