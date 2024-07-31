import numpy as np
from skimage import io, img_as_ubyte
from skimage.color import gray2rgb
import argparse
import os


def load_masks(mask_paths):
    """
    Load PNG masks from the specified paths.
    """
    masks = []
    for path in mask_paths:
        try:
            print(f"Loading mask from {path}")
            mask = io.imread(path, as_gray=True)
            if mask is not None:
                masks.append(mask)
                print(f"Loaded mask shape: {mask.shape}")
            else:
                print(f"Warning: Mask at {path} is None.")
        except Exception as e:
            print(f"Error loading mask from {path}: {e}")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    print(f"Total masks loaded: {len(masks)}")
    return np.stack(masks, axis=-1)


def staple(masks, max_iter=10, tol=1e-5):
    """
    STAPLE algorithm for fusing segmentation masks.
    """
    num_masks = masks.shape[-1]
    print(f"Number of masks: {num_masks}")
    weights = np.ones(num_masks) / num_masks
    consensus = np.mean(masks, axis=-1)

    for iteration in range(max_iter):
        new_weights = np.zeros_like(weights)

        for i in range(num_masks):
            new_weights[i] = np.mean(consensus[masks[..., i] == 1])

        new_weights /= np.sum(new_weights)
        new_consensus = np.sum(masks * new_weights, axis=-1)

        max_diff = np.max(np.abs(new_consensus - consensus))
        print(f"Iteration {iteration}: max difference = {max_diff}")

        if max_diff < tol:
            print("Convergence reached.")
            break

        consensus = new_consensus
        weights = new_weights

    return consensus > 0.5


def calculate_statistics(mask):
    """
    Calculate statistics for a segmentation mask.
    """
    total_pixels = mask.size
    foreground_pixels = np.sum(mask)
    background_pixels = total_pixels - foreground_pixels
    stats = {
        'Total Pixels': total_pixels,
        'Foreground Pixels': foreground_pixels,
        'Background Pixels': background_pixels,
        'Foreground Percentage': foreground_pixels / total_pixels * 100,
        'Background Percentage': background_pixels / total_pixels * 100
    }
    return stats


def save_statistics_report(stats_list, output_file):
    """
    Save statistics report to a text file.
    """
    with open(output_file, 'w') as f:
        for i, stats in enumerate(stats_list):
            f.write(f"Mask {i + 1}:\n")
            for metric, value in stats.items():
                f.write(f"  {metric}: {value:.2f}\n")
            f.write("\n")


def main(mask_paths, output_file_base):
    print("Starting the mask fusion process...")
    masks = load_masks(mask_paths)
    fused_mask = staple(masks)
    print("Mask fusion completed.")

    stats_list = [calculate_statistics(mask) for mask in masks]
    fused_stats = calculate_statistics(fused_mask)
    stats_list.append(fused_stats)

    report_file = output_file_base + '.txt'
    save_statistics_report(stats_list, report_file)
    print(f"Statistics report saved to {report_file}")

    # Save the fused mask as an image
    fused_mask_image = img_as_ubyte(fused_mask)
    fused_mask_image_rgb = gray2rgb(fused_mask_image)
    image_file = output_file_base + '.png'
    io.imsave(image_file, fused_mask_image_rgb)
    print(f"Fused mask image saved to {image_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse segmentation masks using STAPLE and calculate statistics.')
    parser.add_argument('--mask_paths', type=str, nargs='+', required=True, help='Paths to PNG mask files.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Base path to save the fused mask image and statistics report.')

    args = parser.parse_args()
    main(args.mask_paths, args.output_file)
