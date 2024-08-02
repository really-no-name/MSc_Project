import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def display_images_with_overlay(name, path_A, path_B_1, path_B_2, path_B_3, path_B_4, path_B_5, path_B_6, path_C,
                                images_to_display, show, resize_size=(256, 256), ):
    # 加载原始图像
    original_image_path = f"{path_A}{name}.png"
    original_image = Image.open(original_image_path)

    # 调整原始图像大小至resize_size
    original_image = original_image.resize(resize_size)

    # 加载掩码图像
    mask_image_paths = [f"{path_B_1}{name}.png", f"{path_B_2}{name}.png", f"{path_B_3}{name}.png",
                        f"{path_B_4}{name}.png", f"{path_B_5}{name}.png", f"{path_B_6}{name}.png"]
    mask_images = [Image.open(mask_image_path).resize(resize_size).convert("L") for mask_image_path in mask_image_paths]

    # 加载其他图像
    other_image_paths = [f"{path_C}{name}_{suffix}.png" for suffix in images_to_display]
    other_images = [Image.open(other_image_path).resize(resize_size) for other_image_path in other_image_paths]

    # 加载其他图像不确定性
    other_imageUE_paths = [f"{path_C}{name}_{suffix}_UE.png" for suffix in images_to_display]
    other_imagesUE = [Image.open(other_imageUE_path).resize(resize_size) for other_imageUE_path in other_imageUE_paths]

    # 获取图像高度
    _, height = original_image.size

    # 显示图像
    total_images = 1 + len(mask_images) + len(other_images)
    fig, axes = plt.subplots(4, max(len(mask_images), len(other_images), 1), figsize=(20, 20),
                             gridspec_kw={'hspace': 0.1})

    # 显示原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].grid(which='both', color='white', linestyle='-', linewidth=0.5)
    axes[0, 0].tick_params(axis='both', which='both', labelsize=8)

    for i in range(1, len(axes[0])):
        axes[0, i].axis('off')  # 关闭未使用的子图

    # 显示掩码图像
    for idx, mask_image in enumerate(mask_images):
        axes[1, idx].imshow(mask_image, cmap="gray")
        axes[1, idx].set_title(f"Mask Image {idx + 1}", fontsize=12)
        axes[1, idx].grid(which='both', color='white', linestyle='-', linewidth=0.5)
        axes[1, idx].tick_params(axis='both', which='both', labelsize=8)

    for i in range(len(mask_images), len(axes[1])):
        axes[1, i].axis('off')  # 关闭未使用的子图

    # 显示其他图像
    for idx, other_image in enumerate(other_images):
        suffix = images_to_display[idx]
        axes[2, idx].imshow(other_image, cmap="gray")
        axes[2, idx].set_title(f"{suffix} Image", fontsize=12)
        axes[2, idx].grid(which='both', color='white', linestyle='-', linewidth=0.5)
        axes[2, idx].tick_params(axis='both', which='both', labelsize=8)

    for i in range(len(other_images), len(axes[2])):
        axes[2, i].axis('off')  # 关闭未使用的子图

    # 显示其他图像不确定性
    for idx, other_imageUE in enumerate(other_imagesUE):
        suffix = images_to_display[idx]
        axes[3, idx].imshow(other_imageUE, cmap="gray")
        axes[3, idx].set_title(f"{suffix} Uncertainty", fontsize=12)
        axes[3, idx].grid(which='both', color='white', linestyle='-', linewidth=0.5)
        axes[3, idx].tick_params(axis='both', which='both', labelsize=8)

    for i in range(len(other_imagesUE), len(axes[3])):
        axes[3, i].axis('off')  # 关闭未使用的子图

    plt.tight_layout()
    plt.savefig(path_A + name + "_show.png")
    if show == 'on':
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images with overlay')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')
    parser.add_argument('-show', '--show', choices=['on', 'off'], required=True,
                        help='Whether to show the image or not')

    args = parser.parse_args()

    if args.mode == 'train':
        source = "Train_imgs"
    elif args.mode == 'test':
        source = "Test_imgs"

    name = args.name
    path_A = source + "/"
    path_B = "dataset/Gleason19/resized_dataset_1024/Maps/"
    path_B_1 = path_B + "Maps1_T/"
    path_B_2 = path_B + "Maps2_T/"
    path_B_3 = path_B + "Maps3_T/"
    path_B_4 = path_B + "Maps4_T/"
    path_B_5 = path_B + "Maps5_T/"
    path_B_6 = path_B + "Maps6_T/"
    path_C = source + "/"
    images_to_display = ["STAPLE", "MV", "AV"]  # 其他图像的后缀名
    show = args.show

    display_images_with_overlay(name, path_A, path_B_1, path_B_2, path_B_3, path_B_4, path_B_5, path_B_6, path_C,
                                images_to_display, show)
