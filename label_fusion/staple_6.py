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
    class_masks = {}
    for cls in range(5):  # Assuming 5 classes per each of the 6 sets
        class_masks[cls] = [masks[i * 5 + cls] for i in range(6)]
    return class_masks, list(class_masks.keys())


def staple(class_masks, print_state):
    """
    STAPLE algorithm for fusing segmentation masks using SimpleITK.
    """
    uncertainties = {}
    for cls, masks in class_masks.items():
        if print_state == 'on':
            print(f"Processing class {cls} with {len(masks)} masks")

        # Normalize the masks to [0, 1]
        masks = [cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32) for mask in masks]

        # Convert masks to 8-bit unsigned integers for STAPLE processing
        masks = [(mask * 255).astype(np.uint8) for mask in masks]

        masks_sitk = [sitk.GetImageFromArray(mask) for mask in masks]
        staple_filter = sitk.STAPLEImageFilter()
        staple_mask_sitk = staple_filter.Execute(masks_sitk)

        # Calculate pixel-wise uncertainty as the standard deviation across masks
        masks_array = np.array([sitk.GetArrayFromImage(mask) for mask in masks_sitk])
        uncertainty = np.std(masks_array, axis=0)
        uncertainties[cls] = uncertainty

    return uncertainties


def combine_uncertainties(uncertainties, classes):
    """
    Combine uncertainties from different classes into a single map.
    """
    combined_uncertainty = np.zeros_like(next(iter(uncertainties.values())), dtype=np.float32)
    for cls in classes:
        combined_uncertainty += uncertainties[cls]
    combined_uncertainty /= len(classes)
    return combined_uncertainty


def load_masks(mask_paths, print_state):
    """
    Load PNG masks from the specified paths.
    """
    masks = []
    for path in mask_paths:
        if print_state == 'on':
            print(f"Loading mask from {path}")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
            if print_state == 'on':
                print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        else:
            print(f"Warning: Mask at {path} is None or could not be loaded.")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    return np.array(masks)


def save_uncertainty_white(uncertainty, output_path, print_state):
    """
    Save the uncertainty map to the specified output path.
    """
    normalized_uncertainty = cv2.normalize(uncertainty, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, normalized_uncertainty.astype(np.uint8))
    if print_state == 'on':
        print(f"Uncertainty map saved to {output_path}")

def save_uncertainty_black(uncertainty, output_path, print_state):
    """
    Save the uncertainty map to the specified output path.
    """
    normalized_uncertainty = cv2.normalize(uncertainty, None, 0, 255, cv2.NORM_MINMAX)
    inverted_uncertainty = 255 - normalized_uncertainty
    cv2.imwrite(output_path, inverted_uncertainty.astype(np.uint8))
    if print_state == 'on':
        print(f"Uncertainty map saved to {output_path}")


def save_uncertainty_with_legend_white(uncertainty, output_path, print_state):
    """
    Save the uncertainty map with a gray color legend to the specified output path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(uncertainty, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Uncertainty', cmap='gray')
    plt.title('Uncertainty Map')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    if print_state == 'on':
        print(f"Uncertainty map with legend saved to {output_path}")


def save_uncertainty_with_legend_black(uncertainty, output_path, print_state):
    """
    Save the uncertainty map with a gray color legend to the specified output path.
    """
    normalized_uncertainty = cv2.normalize(uncertainty, None, 0, 255, cv2.NORM_MINMAX)
    inverted_uncertainty = 255 - normalized_uncertainty
    plt.figure(figsize=(10, 10))
    plt.imshow(inverted_uncertainty, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Uncertainty', cmap='gray')
    plt.title('Uncertainty Map')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    if print_state == 'on':
        print(f"Uncertainty map with legend saved to {output_path}")


def main_1(mask_paths, uncertainty_output_path, uncertainty_legend_output_path, print_state):
    if print_state == 'on':
        print("Starting the mask fusion process...")

    masks = load_masks(mask_paths, print_state)
    if print_state == 'on':
        print(f"Loaded {len(masks)} masks.")

    class_masks, classes = preprocess_masks(masks)
    if print_state == 'on':
        print(f"Classes found: {classes}")

    uncertainties = staple(class_masks, print_state)

    combined_uncertainty = combine_uncertainties(uncertainties, classes)

    if print_state == 'on':
        print("Mask fusion completed.")
    save_uncertainty_white(combined_uncertainty, uncertainty_output_path, print_state)
    save_uncertainty_with_legend_white(combined_uncertainty, uncertainty_legend_output_path, print_state)

    # Output the range of values in the uncertainty map
    uncertainty_min = np.min(combined_uncertainty)
    uncertainty_max = np.max(combined_uncertainty)
    if print_state == 'on':
        print(f"Uncertainty map value range: min={uncertainty_min}, max={uncertainty_max}")


def main_2(mask_paths, uncertainty_output_path, uncertainty_legend_output_path, print_state):
    if print_state == 'on':
        print("Starting the mask fusion process...")
    masks = load_masks(mask_paths, print_state)
    if print_state == 'on':
        print(f"Loaded {len(masks)} masks.")

    class_masks, classes = preprocess_masks(masks)
    if print_state == 'on':
        print(f"Classes found: {classes}")

    uncertainties = staple(class_masks, print_state)

    combined_uncertainty = combine_uncertainties(uncertainties, classes)

    if print_state == 'on':
        print("Mask fusion completed.")
    save_uncertainty_black(combined_uncertainty, uncertainty_output_path, print_state)
    save_uncertainty_with_legend_black(combined_uncertainty, uncertainty_legend_output_path, print_state)

    # Output the range of values in the uncertainty map
    uncertainty_min = np.min(combined_uncertainty)
    uncertainty_max = np.max(combined_uncertainty)
    if print_state == 'on':
        print(f"Uncertainty map value range: min={uncertainty_min}, max={uncertainty_max}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Staple 模型投票，生成不确定性图像')
    parser.add_argument('-t', '--type', type=str, default='black', choices=['white', 'black'], help='表示高不确定性的颜色')
    parser.add_argument('-p', '--print_state', type=str, default='on', choices=['on', 'off'], help='是否print')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')
    args = parser.parse_args()

    if args.mode == 'train':
        source = 'dataset/Gleason19/label_fusion/Train_imgs'
    elif args.mode == 'test':
        source = 'dataset/Gleason19/label_fusion/Test_imgs'

    name = args.name

    print('------------------------------  ' + name + 'Uncertainty Map ------------------------')

    mask_paths = []
    for i in range(1, 7):
        for cls in range(0, 5):
            mask_paths.append(
                f'dataset/Gleason19/output/Maps/Maps{i}_T/pred_{name}_class{cls}_mean_prob.png')

    uncertainty_output_path = source + '/' + name + '_STAPLE_UE.png'
    uncertainty_legend_output_path = source + '/' + name + '_STAPLE_UE_l.png'

    if args.type == 'white':
        main_1(mask_paths, uncertainty_output_path, uncertainty_legend_output_path, args.print_state)
    elif args.type == 'black':
        main_2(mask_paths, uncertainty_output_path, uncertainty_legend_output_path, args.print_state)
