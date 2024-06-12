import os

# 定义路径
image_path = 'resized_dataset_1024/Crossval1/train'
mask_paths = [
    'resized_dataset_1024/Maps/Maps1_T',
    'resized_dataset_1024/Maps/Maps2_T',
    'resized_dataset_1024/Maps/Maps3_T',
    'resized_dataset_1024/Maps/Maps4_T',
    'resized_dataset_1024/Maps/Maps5_T',
    'resized_dataset_1024/Maps/Maps6_T'
]

# 获取图像文件列表
image_files = set(os.listdir(image_path))

# 获取所有掩码文件列表
mask_files = set()
for mask_path in mask_paths:
    mask_files.update(os.listdir(mask_path))

# 找出不匹配的文件
unmatched_images = image_files - mask_files
unmatched_masks = mask_files - image_files

if unmatched_images:
    print(f"以下图像文件没有对应的掩码文件: {unmatched_images}")
if unmatched_masks:
    print(f"以下掩码文件没有对应的图像文件: {unmatched_masks}")
if not unmatched_images and not unmatched_masks:
    print("所有图像和掩码文件匹配正确。")
