import os
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms
from tqdm import tqdm


reference_path = "data/reference/00000039_brain_t1.nii"
ref_data = nib.load(reference_path).get_fdata()

modalities = {
    "t1n": "-t1n.nii",
    "t1c": "-t1c.nii",
    "t2w": "-t2w.nii",
    "t2f": "-t2f.nii",
}

# --- Histogram Matching Function ---
def histogram_match(volume, reference):
    return match_histograms(volume, reference, channel_axis=None)



def getitem(image, ref_affine, tranform=None):
    matched_array = histogram_match(image, ref_data)
    # Load modalities
    modality_map = {
        't1': 't1n',
        't1ce': 't1c',
        't2': 't2w',
        'flair': 't2f'
    }


    # Stack modalities: [flair, t1, t1ce, t2] - standard order
    vol = np.stack([
        modalities['flair'][0], 
        modalities['t1'][0], 
        modalities['t1ce'][0], 
        modalities['t2'][0]
    ], axis=0)
    vol_tensor = torch.from_numpy(vol.astype(np.float32))

    # Create TorchIO subject with only image (no segmentation)
    image = tio.ScalarImage(tensor=vol_tensor, affine=ref_affine)
    subject = tio.Subject(image=image)

    # Apply basic preprocessing
    subject = tio.Compose([
        tio.ToCanonical(),
        tio.Resample((1, 1, 1)),
    ])(subject)

    if transforms:
        subject = transforms(subject)

    # Return only image data and subject ID
    return {
        'image': subject['image'].data,
        'subject_id': data_id
        }
