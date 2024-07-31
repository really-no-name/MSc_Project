import argparse
import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt


def preprocess_mask(mask):
    """
    Preprocess the mask by converting specific values to binary (0 and 1).
    """
    processed_mask = np.where(mask == 4, 1, 0)
    return processed_mask


def staple(masks):
    """
    STAPLE algorithm for fusing segmentation masks using SimpleITK.
    """
    masks_sitk = [sitk.GetImageFromArray(mask.astype(np.uint8)) for mask in masks]
    staple_filter = sitk.STAPLEImageFilter()
    staple_mask_sitk = staple_filter.Execute(masks_sitk)
    staple_mask = sitk.GetArrayFromImage(staple_mask_sitk)

    # Calculate pixel-wise uncertainty as the standard deviation across masks
    masks_array = np.array([sitk.GetArrayFromImage(mask) for mask in masks_sitk])
    uncertainty = np.std(masks_array, axis=0)

    return staple_mask > 0.5, uncertainty


def load_masks(mask_paths):
    """
    Load PNG masks from the specified paths.
    """
    masks = []
    for path in mask_paths:
        print(f"Loading mask from {path}")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            processed_mask = preprocess_mask(mask)
            masks.append(processed_mask)
            print(
                f"Mask shape: {mask.shape}, unique values before processing: {np.unique(mask)}, unique values after processing: {np.unique(processed_mask)}")
        else:
            print(f"Warning: Mask at {path} is None or could not be loaded.")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    return np.array(masks)


def save_fused_mask(fused_mask, output_path):
    """
    Save the fused mask to the specified output path.
    """
    cv2.imwrite(output_path, (fused_mask * 255).astype(np.uint8))
    print(f"Fused mask saved to {output_path}")


def save_uncertainty(uncertainty, output_path):
    """
    Save the uncertainty map to the specified output path.
    """
    normalized_uncertainty = cv2.normalize(uncertainty, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, normalized_uncertainty.astype(np.uint8))
    print(f"Uncertainty map saved to {output_path}")


def save_uncertainty_with_legend(uncertainty, output_path):
    """
    Save the uncertainty map with a gray color legend to the specified output path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(uncertainty, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Uncertainty', cmap='gray')
    plt.title('Uncertainty Map')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Uncertainty map with legend saved to {output_path}")


def main(mask_paths, fused_output_path, uncertainty_output_path, uncertainty_legend_output_path):
    print("Starting the mask fusion process...")
    masks = load_masks(mask_paths)
    print(f"Loaded {len(masks)} masks.")

    if masks.size == 0:
        raise ValueError("No masks were loaded. Please check the input paths.")

    fused_mask, uncertainty = staple(masks)
    print(f"Fused mask unique values: {np.unique(fused_mask)}")

    if np.all(fused_mask == 0):
        print("Warning: The fused mask is completely black.")

    print("Mask fusion completed.")
    save_fused_mask(fused_mask, fused_output_path)
    save_uncertainty(uncertainty, uncertainty_output_path)
    save_uncertainty_with_legend(uncertainty, uncertainty_legend_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Staple 模型投票，二元分类')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')

    args = parser.parse_args()

    name = args.name

    mask_paths = [
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps1_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps2_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps3_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps4_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps5_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps6_T/' + name + '.png'
    ]

    fused_output_path = 'test/' + name + '_STAPLE.png'
    uncertainty_output_path = 'test/' + name + '_STAPLE_UE.png'
    uncertainty_legend_output_path = 'test/' + name + '_STAPLE_UE_l.png'

    main(mask_paths, fused_output_path, uncertainty_output_path, uncertainty_legend_output_path)
