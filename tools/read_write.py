'''
Placeholders
'''
import torch
import os
import nibabel as nib
import numpy as np
from tools.torch_stuff import get_segformer3d_plus


def load_nii(img_pth: str):
    # Load reference to get original shape
    # reference_path = os.path.join(validation_dir, subject_id, f"{subject_id}-t1n.nii")
    img = nib.load(img_pth)
    affine, header =  img.affine, img.header
    img_array = img.get_fdata()
    ref_shape = image.shape
    return img_array, affine, header

def save_nii(predicted_seg, save_path, affine, header):
    # seg = nib.Nifti1Image(predicted_seg, affine=affine, header=header)
    # nib.save(seg, save_path)
    # Save final segmentation
    # Place prediction in center region
    # Create full-size prediction volume (initialize with background)
    # Calculate center crop coordinates to place 96x96x96 prediction
    # This assumes your CropOrPad centered the crop
    start_x = (ref_shape[0] - 96) // 2
    start_y = (ref_shape[1] - 96) // 2  
    start_z = (ref_shape[2] - 96) // 2
    
    end_x = start_x + 96
    end_y = start_y + 96
    end_z = start_z + 96
    full_prediction = np.zeros(ref_shape, dtype=np.uint8)
    full_prediction[start_x:end_x, start_y:end_y, start_z:end_z] = prediction_seg

    out_path = os.path.join(save_path, f"{subject_id}.nii.gz")
    nib.save(nib.Nifti1Image(full_prediction, affine=affine, header=header), out_path)
    return None