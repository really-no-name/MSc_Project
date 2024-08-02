import argparse
import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt


def preprocess_masks(masks):
    """
    Preprocess the masks by separating them into binary masks for each class.
    """
    classes = np.unique(masks)
    class_masks = {cls: (masks == cls).astype(np.uint8) for cls in classes}
    return class_masks, classes


def staple(class_masks):
    """
    STAPLE algorithm for fusing segmentation masks using SimpleITK.
    """
    fused_class_masks = {}
    uncertainties = {}
    for cls, masks in class_masks.items():
        masks_sitk = [sitk.GetImageFromArray(mask) for mask in masks]
        staple_filter = sitk.STAPLEImageFilter()
        staple_mask_sitk = staple_filter.Execute(masks_sitk)
        staple_mask = sitk.GetArrayFromImage(staple_mask_sitk)
        fused_class_masks[cls] = staple_mask

        # Calculate pixel-wise uncertainty as the standard deviation across masks
        masks_array = np.array([sitk.GetArrayFromImage(mask) for mask in masks_sitk])
        uncertainty = np.std(masks_array, axis=0)
        uncertainties[cls] = uncertainty

    return fused_class_masks, uncertainties


def combine_fused_masks(fused_class_masks, classes):
    """
    Combine fused binary masks into a single multi-class mask.
    """
    combined_mask = np.zeros_like(next(iter(fused_class_masks.values())), dtype=np.uint8)
    for cls in classes:
        combined_mask[fused_class_masks[cls] > 0.5] = cls
    return combined_mask


def combine_uncertainties(uncertainties, classes):
    """
    Combine uncertainties from different classes into a single map.
    """
    combined_uncertainty = np.zeros_like(next(iter(uncertainties.values())), dtype=np.float32)
    for cls in classes:
        combined_uncertainty += uncertainties[cls]
    combined_uncertainty /= len(classes)
    return combined_uncertainty


def load_masks(mask_paths):
    """
    Load PNG masks from the specified paths.
    """
    masks = []
    for path in mask_paths:
        print(f"Loading mask from {path}")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
            print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        else:
            print(f"Warning: Mask at {path} is None or could not be loaded.")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    return np.array(masks)


def save_fused_mask(fused_mask, output_path):
    """
    Save the fused mask to the specified output path.
    """
    cv2.imwrite(output_path, fused_mask)
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

    class_masks, classes = preprocess_masks(masks)
    print(f"Classes found: {classes}")

    fused_class_masks, uncertainties = staple(class_masks)

    fused_mask = combine_fused_masks(fused_class_masks, classes)
    combined_uncertainty = combine_uncertainties(uncertainties, classes)

    print(f"Fused mask unique values: {np.unique(fused_mask)}")

    if np.all(fused_mask == 0):
        print("Warning: The fused mask is completely black.")

    print("Mask fusion completed.")
    save_fused_mask(fused_mask, fused_output_path)
    save_uncertainty(combined_uncertainty, uncertainty_output_path)
    save_uncertainty_with_legend(combined_uncertainty, uncertainty_legend_output_path)

    # Output the range of values in the uncertainty map
    uncertainty_min = np.min(combined_uncertainty)
    uncertainty_max = np.max(combined_uncertainty)
    print(f"Uncertainty map value range: min={uncertainty_min}, max={uncertainty_max}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Staple 模型投票，0-4分类')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'], help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')
    args = parser.parse_args()

    if args.mode == 'train':
        source = 'Train_imgs'
    elif args.mode == 'test':
        source = 'Test_imgs'

    name = args.name

    mask_paths = [
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps1_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps2_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps3_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps4_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps5_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps6_T/' + name + '.png'
    ]

    fused_output_path = source + '/' + name + '_STAPLE.png'
    uncertainty_output_path = source + '/' + name + '_STAPLE_UE.png'
    uncertainty_legend_output_path = source + '/' + name + '_STAPLE_UE_l.png'

    main(mask_paths, fused_output_path, uncertainty_output_path, uncertainty_legend_output_path)
