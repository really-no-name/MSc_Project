import argparse
import os
from PIL import Image
import numpy as np


def get_image_details(directory):
    """
    读取指定文件夹下的所有PNG格式掩码图像，返回每张图像中的所有唯一值和左上角第一个像素的值。

    参数:
    directory (str): 图像文件夹的路径

    返回:
    dict: 包含每张图像的所有唯一值和左上角第一个像素值的字典
    """
    image_details = {}

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)
            image_array = np.array(image)

            # 获取图像中的所有唯一值
            unique_values = np.unique(image_array)
            # 获取左上角第一个像素的值
            top_left_pixel_value = image_array[0, 0]

            image_details[filename] = {
                'unique_values': unique_values,
                'top_left_pixel': top_left_pixel_value
            }

    return image_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images with overlay')
    parser.add_argument('-p', '--path', type=str, required=True, help='The path to your directory')

    args = parser.parse_args()

    directory = args.path  # 将这里替换为你的文件夹路径
    image_details_dict = get_image_details(directory)

    # 打印每张图像的所有唯一值和左上角第一个像素值
    for image_name, details in image_details_dict.items():
        unique_values = details['unique_values']
        top_left_pixel = details['top_left_pixel']
        print(f"Image: {image_name}, Unique Values: {unique_values}, Top Left Pixel: {top_left_pixel}")
