import argparse
import os
from PIL import Image
import numpy as np


def get_image_unique_values(directory):
    """
    读取指定文件夹下的所有PNG格式掩码图像，返回每张图像中的所有唯一值。

    参数:
    directory (str): 图像文件夹的路径

    返回:
    dict: 包含每张图像的所有唯一值的字典
    """
    unique_values = {}

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)
            image_array = np.array(image)

            # 获取图像中的所有唯一值
            values = np.unique(image_array)

            unique_values[filename] = values

    return unique_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images with overlay')
    parser.add_argument('-p', '--path', type=str, required=True, help='The path to your directory')

    args = parser.parse_args()

    directory = args.path  # 将这里替换为你的文件夹路径
    unique_values_dict = get_image_unique_values(directory)

    # 打印每张图像的所有唯一值
    for image_name, values in unique_values_dict.items():
        print(f"Image: {image_name}, Unique Values: {values}")
