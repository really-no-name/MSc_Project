from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def display_images_with_overlay(original_image_path, mask_image_path, resize_size=(256, 256)):
    # 加载图像
    original_image = Image.open(original_image_path)
    mask_image = Image.open(mask_image_path)

    # 调整两张图像大小至resize_size
    original_image = original_image.resize(resize_size)
    mask_image = mask_image.resize(resize_size)

    # 获取图像高度
    _, height = mask_image.size

    # 确保两张图像都处于相同的模式和大小
    original_image = original_image.convert("RGBA")
    mask_image = mask_image.convert("L")  # 将掩码转换为灰度图像

    # 创建一个带有掩码区域特定颜色的 RGBA 版本的掩码
    mask_rgba = Image.new("RGBA", mask_image.size)
    mask_data = np.array(mask_image)
    mask_rgba_data = np.array(mask_rgba)

    # 定义掩码叠加颜色
    overlay_color = [255, 0, 0, 150]  # 红色带透明度

    # 将叠加颜色应用于掩码区域
    mask_rgba_data[mask_data > 0] = overlay_color
    mask_rgba = Image.fromarray(mask_rgba_data, "RGBA")

    # 将原始图像与掩码混合
    blended_image = Image.alpha_composite(original_image, mask_rgba)

    # 显示图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].set_xticks(np.arange(0, original_image.size[0], step=height/4))
    axes[0].set_yticks(np.arange(0, original_image.size[1], step=height/4))
    axes[0].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    axes[0].tick_params(axis='both', which='both', labelsize=8)

    # 显示掩码图像
    axes[1].imshow(mask_image, cmap="gray")
    axes[1].set_title("Mask Image")
    axes[1].set_xticks(np.arange(0, mask_image.size[0], step=height/4))
    axes[1].set_yticks(np.arange(0, mask_image.size[1], step=height/4))
    axes[1].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    axes[1].tick_params(axis='both', which='both', labelsize=8)

    # 显示叠加图像
    axes[2].imshow(blended_image)
    axes[2].set_title("Overlay Image")
    axes[2].set_xticks(np.arange(0, blended_image.size[0], step=height/4))
    axes[2].set_yticks(np.arange(0, blended_image.size[1], step=height/4))
    axes[2].grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    axes[2].tick_params(axis='both', which='both', labelsize=8)

    plt.tight_layout()
    plt.show()