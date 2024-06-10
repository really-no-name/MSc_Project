from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_images(tissue_path, mask_path):
    # Load the images
    tissue_image = Image.open(tissue_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")  # Convert to grayscale

    return tissue_image, mask_image

def overlay_images(tissue_image, mask_image):
    # Convert images to numpy arrays
    tissue_array = np.array(tissue_image)
    mask_array = np.array(mask_image)

    # Create a red mask
    red_mask = np.zeros_like(tissue_array)
    red_mask[:, :, 0] = mask_array  # Set the red channel where the mask is non-zero
    red_mask[:, :, 1] = 0  # Green channel
    red_mask[:, :, 2] = 0  # Blue channel

    # Combine tissue image with red mask
    combined_image = tissue_array.copy()
    combined_image[mask_array > 0] = red_mask[mask_array > 0]  # Apply the red mask to the tissue image

    return combined_image

def display_images(tissue_image, mask_image, combined_image):
    # Display the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tissue_image)
    axes[0].set_title("Tissue Image")
    axes[0].axis('off')

    axes[1].imshow(mask_image, cmap='gray')
    axes[1].set_title("Mask Image")
    axes[1].axis('off')

    axes[2].imshow(combined_image)
    axes[2].set_title("Combined Image")
    axes[2].axis('off')

    plt.show()

# Paths to the images
tissue_path = "/Volumes/Ugreen/study/MSc Project/Dataset/Gleason_2019/Train Imgs/slide001_core003.jpg"
mask_path = "/Volumes/Ugreen/study/MSc Project/Dataset/Gleason_2019/Maps1_T/slide001_core003_classimg_nonconvex.png"

# Check if paths are correct
import os
print(f"Tissue path exists: {os.path.exists(tissue_path)}")
print(f"Mask path exists: {os.path.exists(mask_path)}")

# Load the images
tissue_image, mask_image = load_images(tissue_path, mask_path)

# Overlay the images
combined_image = overlay_images(tissue_image, mask_image)

# Display the images
display_images(tissue_image, mask_image, combined_image)