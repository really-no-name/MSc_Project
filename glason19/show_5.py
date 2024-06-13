## 将组织切片图像和掩码图像叠加显示，以方便观察
## 使用不同颜色对掩码图像中不同的值进行标注，对应Gleason Grade 的不同级别


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the images
file_name = "slide002_core002.png"
original_image_path = "resized_dataset_1024/train/" + file_name
mask_image_path = "resized_dataset_1024/Maps/Maps2_T/" + file_name

original_image = Image.open(original_image_path)
mask_image = Image.open(mask_image_path)

# Ensure both images are in the same mode and size
original_image = original_image.convert("RGBA")
mask_image = mask_image.convert("L")  # Convert mask to grayscale

# Create an RGBA version of the mask with specific colors for different mask values
mask_rgba = Image.new("RGBA", mask_image.size)
mask_data = np.array(mask_image)
mask_rgba_data = np.array(mask_rgba)

# Define overlay colors for mask values
# overlay_colors = {
#     0: [0, 0, 0, 0],       # Transparent for 0
#     1: [255, 0, 0, 150],   # Red for 1
#     2: [0, 255, 0, 150],   # Green for 2
#     3: [0, 0, 255, 150],   # Blue for 3 -- Gleason grade 3
#     4: [255, 255, 0, 150], # Yellow for 4 -- Gleason grade 4
#     5: [0, 255, 255, 150], # Cyan for 5 -- Gleason grade 5
#     6: [255, 0, 255, 150], # Magenta for 6 -- 分级4+3的病灶（主要成分Gleason等级4，次要成分Gleason等级3）
#     7: [128, 0, 128, 150], # Purple for 7 -- 分级3+4的病灶（主要成分Gleason等级3，次要成分Gleason等级4）
#     8: [128, 128, 0, 150], # Olive for 8 -- 其他特殊情况（如分级5+4或分级4+5）
# }

overlay_colors = {
    0: [0, 0, 0, 0],       # Transparent for 0
    1: [0, 255, 0, 150],   # Green for 1
    2: [0, 255, 0, 150],   # Green for 2
    3: [0, 0, 255, 150],   # Blue for 3 -- Gleason grade 3  65, 72, 196
    4: [255, 255, 0, 150], # Yellow for 4 -- Gleason grade 4
    5: [255, 0, 0, 150], # Red for 5 -- Gleason grade 5
    6: [255, 97, 0, 150], # Yellow for 6 -- 分级4+3的病灶（主要成分Gleason等级4，次要成分Gleason等级3）
    7: [25, 25, 112, 150], # Blue for 7 -- 分级3+4的病灶（主要成分Gleason等级3，次要成分Gleason等级4）
    8: [128, 128, 0, 150], # Olive for 8 -- 其他特殊情况（如分级5+4或分级4+5）
}

# Apply overlay colors based on mask values
for value, color in overlay_colors.items():
    mask_rgba_data[mask_data == value] = color

mask_rgba = Image.fromarray(mask_rgba_data, "RGBA")

# Blend the original image with the mask
blended_image = Image.alpha_composite(original_image, mask_rgba)

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

# Display the images
fig, axes = plt.subplots(1, 4, figsize=(17, 5))
# Display original image
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].set_xticks(np.arange(0, original_image.size[0], step=original_image.size[1]/4))
axes[0].set_yticks(np.arange(0, original_image.size[1], step=original_image.size[1]/4))
axes[0].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
axes[0].tick_params(axis='both', which='both', labelsize=8)

# Display mask image
axes[1].imshow(mask_image, cmap="gray")
axes[1].set_title("Mask Image")
axes[1].set_xticks(np.arange(0, mask_image.size[0], step=mask_image.size[1]/4))
axes[1].set_yticks(np.arange(0, mask_image.size[1], step=mask_image.size[1]/4))
axes[1].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
axes[1].tick_params(axis='both', which='both', labelsize=8)

# Display overlay image
axes[2].imshow(blended_image)
axes[2].set_title("Overlay Image")
axes[2].set_xticks(np.arange(0, blended_image.size[0], step=blended_image.size[1]/4))
axes[2].set_yticks(np.arange(0, blended_image.size[1], step=blended_image.size[1]/4))
axes[2].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
axes[2].tick_params(axis='both', which='both', labelsize=8)

# Add legend
legend_colors = [color[:3] for value, color in overlay_colors.items() if value != 0]
legend_texts = [legend_labels[value] for value in overlay_colors.keys() if value != 0]
patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color)/255, markersize=10) for color in legend_colors]
axes[3].legend(patches, legend_texts, loc='center')
axes[3].axis('off')

plt.tight_layout()
plt.show()