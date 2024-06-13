## 将组织切片图像和六张掩码图像叠加生成，以方便观察

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define paths
original_images_folder = "resized_dataset_1024/train/"
mask_images_folders = [f"resized_dataset_1024/Maps/Maps{i}_T/" for i in range(1, 7)]
output_folder = "generated_images/SixMaps"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define overlay colors for mask values
overlay_colors = {
    0: [0, 0, 0, 0],       # Transparent for 0
    1: [0, 255, 0, 150],   # Green for 1
    2: [0, 255, 0, 150],   # Green for 2
    3: [0, 0, 255, 150],   # Blue for 3 -- Gleason grade 3  65, 72, 196
    4: [255, 255, 0, 150], # Yellow for 4 -- Gleason grade 4
    5: [255, 0, 0, 150],   # Red for 5 -- Gleason grade 5
    6: [255, 97, 0, 150],  # Orange for 6 -- 分级4+3的病灶（主要成分Gleason等级4，次要成分Gleason等级3）
    7: [25, 25, 112, 150], # Dark blue for 7 -- 分级3+4的病灶（主要成分Gleason等级3，次要成分Gleason等级4）
    8: [128, 128, 0, 150], # Olive for 8 -- 其他特殊情况（如分级5+4或分级4+5）
}

# Define custom labels for the legend
legend_labels = {
    1: "Gleason grade 1 -- Benign",
    2: "Gleason grade 2 -- Benign",
    3: "Gleason grade 3",
    4: "Gleason grade 4",
    5: "Gleason grade 5",
    6: "Gleason grade 6 -- graded 4+3 (major grade 4)",
    7: "Gleason grade 7 -- graded 3+4 (major grade 3)",
    8: "Gleason grade 8 -- Other special cases"
}

# Process each file in the original images folder
for file_name in os.listdir(original_images_folder):
    original_image_path = os.path.join(original_images_folder, file_name)
    save_path = os.path.join(output_folder, file_name)

    # Load the original image
    original_image = Image.open(original_image_path).convert("RGBA")

    # Create a figure with 4 columns and enough rows for the masks
    fig, axes = plt.subplots(6, 4, figsize=(20, 30))

    # Display original image in the first column, first row
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    for i in range(1, 6):
        axes[i, 0].axis('off')  # Hide other rows in the first column

    # Initialize lists for masks and blended images
    mask_images = []
    blended_images = []

    for mask_folder in mask_images_folders:
        mask_image_path = os.path.join(mask_folder, file_name)

        if os.path.exists(mask_image_path):
            mask_image = Image.open(mask_image_path).convert("L")  # Convert mask to grayscale
            mask_images.append(mask_image)

            # Create an RGBA version of the mask with specific colors for different mask values
            mask_rgba = Image.new("RGBA", mask_image.size)
            mask_data = np.array(mask_image)
            mask_rgba_data = np.array(mask_rgba)

            # Apply overlay colors based on mask values
            for value, color in overlay_colors.items():
                mask_rgba_data[mask_data == value] = color

            mask_rgba = Image.fromarray(mask_rgba_data, "RGBA")

            # Blend the original image with the mask
            blended_image = Image.alpha_composite(original_image, mask_rgba)
            blended_images.append(blended_image)
        else:
            mask_images.append(None)
            blended_images.append(None)

    # Display mask images in the second column
    for i in range(6):
        if mask_images[i]:
            axes[i, 1].imshow(mask_images[i], cmap="gray")
        else:
            axes[i, 1].imshow(np.zeros_like(np.array(original_image)), cmap="gray")
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Mask Image {i+1}")

    # Display blended images in the third column
    for i in range(6):
        if blended_images[i]:
            axes[i, 2].imshow(blended_images[i])
        else:
            axes[i, 2].imshow(np.zeros_like(np.array(original_image)))
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Blended Image {i+1}")

    # Add legend in the fourth column
    legend_colors = [color[:3] for value, color in overlay_colors.items() if value != 0]
    legend_texts = [legend_labels[value] for value in overlay_colors.keys() if value != 0]
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color)/255, markersize=10) for color in legend_colors]
    axes[0, 3].legend(patches, legend_texts, loc='center')
    axes[0, 3].axis('off')

    for i in range(1, 6):
        axes[i, 3].axis('off')  # Hide other rows in the fourth column

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Processed and saved: {file_name}")