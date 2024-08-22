import argparse
import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]


def preprocess_masks(masks):
    """
    Preprocess the masks by separating them into binary masks for each class.
    """
    class_masks = {}
    for cls in range(5):  # Assuming 5 classes per each of the 6 sets
        class_masks[cls] = [masks[i * 5 + cls] for i in range(6)]
    return class_masks, list(class_masks.keys())


def staple(class_masks):
    """
    STAPLE algorithm for fusing segmentation masks using SimpleITK.
    """
    fused_class_masks = {}
    uncertainties = {}
    for cls, masks in class_masks.items():
        print(f"Processing class {cls} with {len(masks)} masks")

        # Normalize the masks to [0, 1]
        masks = [cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32) for mask in masks]

        # Convert masks to 8-bit unsigned integers for STAPLE processing
        masks = [(mask * 255).astype(np.uint8) for mask in masks]

        # Debug: Check unique values in each normalized mask
        for i, mask in enumerate(masks):
            unique_values = np.unique(mask)
            print(f"Normalized mask {i} for class {cls} unique values: {unique_values}")

        masks_sitk = [sitk.GetImageFromArray(mask) for mask in masks]
        staple_filter = sitk.STAPLEImageFilter()
        staple_mask_sitk = staple_filter.Execute(masks_sitk)
        staple_mask = sitk.GetArrayFromImage(staple_mask_sitk)
        fused_class_masks[cls] = staple_mask

        # Debug: Check unique values in the fused mask for this class
        unique_values = np.unique(fused_class_masks[cls])
        print(f"Fused mask for class {cls} unique values: {unique_values}")

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
        combined_mask[fused_class_masks[cls] > 127] = cls  # Using 127 as the threshold for 8-bit images
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
            # print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        else:
            print(f"Warning: Mask at {path} is None or could not be loaded.")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    return np.array(masks)


def save_fused_mask(fused_mask, output_path):
    """
    Save the fused mask to the specified output path.
    """
    color_mask = np.zeros((fused_mask.shape[0], fused_mask.shape[1], 3), dtype=np.uint8)
    for cls, color in enumerate(CLASS_COLORS_BGR):
        color_mask[fused_mask == cls] = color

    cv2.imwrite(output_path, color_mask)
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

    for cls, fused_mask in fused_class_masks.items():
        save_fused_mask(fused_mask, f"{fused_output_path}_class{cls}.png")

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
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')
    args = parser.parse_args()

    if args.mode == 'train':
        source = '/Users/Google_Drive/dataset/Gleason19/label_fusion/Train_imgs'
    elif args.mode == 'test':
        source = '/Users/Google_Drive/dataset/Gleason19/label_fusion/Test_imgs'

    name = args.name

    mask_paths = []
    for i in range(1, 7):
        for cls in range(0, 5):
            mask_paths.append(
                f'/Users/Google_Drive/dataset/Gleason19/output/Maps/Maps{i}_T/pred_{name}_class{cls}_mean_prob.png')

    fused_output_path = source + '/' + name + '_STAPLE.png'
    uncertainty_output_path = source + '/' + name + '_STAPLE_UE.png'
    uncertainty_legend_output_path = source + '/' + name + '_STAPLE_UE_l.png'

    main(mask_paths, fused_output_path, uncertainty_output_path, uncertainty_legend_output_path)
