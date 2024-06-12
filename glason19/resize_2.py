## 修改图像尺寸，不改变原始比例，并填充以满足正方形

import os
from PIL import Image, ImageOps


def resize_and_pad_image(input_folder, output_folder, size=(1024, 1024)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # 忽略以 '._' 开头的文件
        if filename.startswith('._'):
            continue

        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with Image.open(input_path) as img:
                    if img.mode in ("L", "1"):  # 处理灰度图像和二值图像
                        color = 0  # 黑色填充
                    else:
                        color = (255, 255, 255)  # 彩色图像的白色填充

                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    delta_w = size[0] - img.size[0]
                    delta_h = size[1] - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=color)

                    # 处理文件名
                    base_name, _ = os.path.splitext(filename)
                    parts = base_name.split('_')
                    if len(parts) > 2:
                        new_base_name = '_'.join(parts[:2])
                    else:
                        new_base_name = base_name

                    output_filename = new_base_name + ".png"
                    output_path = os.path.join(output_folder, output_filename)

                    new_img.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_folder = "Source/Maps6_T"  # 输入文件夹的路径
    output_folder = "resized_dataset_1024/Maps/Maps6_T"  # 输出文件夹的路径
    resize_and_pad_image(input_folder, output_folder)