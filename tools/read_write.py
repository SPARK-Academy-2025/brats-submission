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
    return img_array, affine, header

def save_nii(predicted_seg, save_path, affine, header):
    # Simple save without reconstruction to original size
    seg = nib.Nifti1Image(predicted_seg, affine=affine, header=header)
    nib.save(seg, save_path)
    return None