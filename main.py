import argparse
import os
import numpy as np
from tools.read_write import save_nii, load_nii
from tools.inference import segment_image
from tools.preprocessing import preprocess_item


def main(input_dir: str, output_dir: str):
    '''
    Main inference pipeline for brain tumor segmentation.
    '''
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for subject_path in os.listdir(input_dir):
        # Skip files, only process directories
        subject_full_path = os.path.join(input_dir, subject_path)
        if not os.path.isdir(subject_full_path):
            continue
            
        img_paths = {
            't1n': f'{subject_path}-t1n.nii.gz',
            't1c': f'{subject_path}-t1c.nii.gz',
            't2w': f'{subject_path}-t2w.nii.gz',
            't2f': f'{subject_path}-t2f.nii.gz',
        }
        img_paths = {contrast: os.path.join(input_dir,subject_path, pth) for contrast, pth in img_paths.items()}
        
        # Check if all required files exist
        missing_files = [pth for pth in img_paths.values() if not os.path.exists(pth)]
        if missing_files:
            print(f"Warning: Missing files for subject {subject_path}: {missing_files}")
            continue
        
        # Get affine and header info
        _, affine, header = load_nii(img_paths['t1n'])
        
        try:
            predicted_seg = infer_one_subject(img_paths=img_paths)
            
            save_path = os.path.join(output_dir, f'{subject_path}.nii.gz')
            save_nii(predicted_seg, save_path, affine, header)
            print(f"Saved prediction to {save_path}")
            
        except Exception as e:
            print(f"Error processing subject {subject_path}: {str(e)}")
            continue


def infer_one_subject(img_paths: dict):
    print(f'Running inference using: {img_paths}')
    
    # Load images and get affine from first image
    images = {}
    affine = None
    
    for contrast, pth in img_paths.items():
        img_data, img_affine, _ = load_nii(pth)
        images[contrast] = img_data
        if affine is None:
            affine = img_affine
    
    # Process images (preprocessing)
    processed_image = preprocess_item(images, affine)    
    
    # Run segmentation
    prediction = segment_image(processed_image)
    
    return prediction  # segment_image already does post-processing


def parse_args():
    parser = argparse.ArgumentParser(description="Run the main processing pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to input directory")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(input_dir=args.input, output_dir=args.output)
