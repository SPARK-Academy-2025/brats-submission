import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
from tqdm import tqdm
import torchio as tio
import torch


reference_path = "data/reference/00000039_brain_t1.nii"

# Load reference data with error handling
if os.path.exists(reference_path):
    ref_data = nib.load(reference_path).get_fdata()
else:
    ref_data = None
    print(f"Warning: Reference file not found at {reference_path}")

test_transforms = tio.Compose([
    tio.ZNormalization(include=['image']),
    tio.CropOrPad((96, 96, 96)),
])


# --- Histogram Matching Function ---
def histogram_match(volume):
    if ref_data is not None:
        return match_histograms(volume, reference=ref_data, channel_axis=None)
    else:
        return volume  # Return original if no reference available



def preprocess_item(images, affine, transforms=test_transforms):
    """
    Modified getitem that accepts affine information.
    """
    # Apply histogram matching to each modality
    matched_images = {}
    for modality, img_data in images.items():
        matched_images[modality] = histogram_match(img_data)
    
    # Stack modalities in standard order: [flair, t1, t1ce, t2]
    modality_mapping = {
        'flair': 't2f',
        't1': 't1n',
        't1ce': 't1c', 
        't2': 't2w'
    }
    
    vol = np.stack([
        matched_images[modality_mapping['flair']],
        matched_images[modality_mapping['t1']],
        matched_images[modality_mapping['t1ce']],
        matched_images[modality_mapping['t2']]
    ], axis=0)
    
    vol_tensor = torch.from_numpy(vol.astype(np.float32))
    
    # Use the provided affine
    image = tio.ScalarImage(tensor=vol_tensor, affine=affine)
    subject = tio.Subject(image=image)
    
    # Apply preprocessing
    subject = tio.Compose([
        tio.ToCanonical(),
        tio.Resample((1, 1, 1)),
    ])(subject)
    
    if transforms:
        subject = transforms(subject)
    
    return subject['image'].data
