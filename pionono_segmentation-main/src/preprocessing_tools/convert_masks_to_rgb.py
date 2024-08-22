# 处理Gleason19前列腺TMA（组织微阵列）图像

import argparse
import os
import cv2
import shutil
import random
import numpy as np
from scipy import stats
from utils.saving import CLASS_COLORS_BGR

# ----------------------------------------------------------
# 设置命令行参数：
# 定义了两个命令行参数--input_dir和--output_dir，分别指定输入数据集目录和输出目录，并提供默认值
parser = argparse.ArgumentParser(description="Resize Images of prostate TMA")  # description参数提供了对该程序的简短描述。当用户使用--help选项运行程序时，这个描述将会显示，帮助用户理解这个程序的用途。
# parser.add_argument("--input_dir", "-i", type=str,
#                     default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/",
#                     help="Input directory of dataset.")
# parser.add_argument("--output_dir", "-o", type=str,
#                     default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/rgb_masks/",
#                     help="Output directory of converted masks.")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/content/drive/Othercomputers/Mac/Google_Drive/dataset/Gleason19/resized_dataset_1024/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/content/drive/Othercomputers/Mac/Google_Drive/dataset/Gleason19/resized_dataset_1024/rgb_masks/",
                    help="Output directory of converted masks.")
args = parser.parse_args()

# -------------------------------------------------------------------------
# 定义需要处理的目录：
map_dir = 'Maps/'
map_annotator_dirs = ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/', 'STAPLE/', 'MV/']

# -------------------------------------------------------------------------
# 创建输出目录：
# 如果输出目录不存在，则创建它
os.makedirs(args.output_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 遍历每个标注目录并处理其中的图像：
# 这个循环遍历每个标注目录，读取图像文件列表并逐个处理：
# 	•	生成输入和输出图像的完整路径。
# 	•	使用cv2.imread读取图像。
# 	•	打印图像文件名和其形状信息。
# 	•	定义一个与图像大小相同的全为1的数组ones。
# 	•	根据类颜色映射（CLASS_COLORS_BGR），将图像中对应类的像素值替换为相应的颜色。
# 	•	断言图像的所有像素值在0到255之间，以确保像素值合法。
# 	•	使用cv2.imwrite保存处理后的图像到输出路径。
for m in range(len(map_annotator_dirs)):
    map_annotator_dir = map_annotator_dirs[m]
    in_dir = args.input_dir + map_dir + map_annotator_dir
    out_dir = args.output_dir + map_dir + map_annotator_dir
    os.makedirs(out_dir, exist_ok=True)
    img_file_list = os.listdir(in_dir)
    print(map_annotator_dir)
    print('Images found:' + str(len(img_file_list)))
    resolution_list = []
    for img_file in img_file_list:
        img_path_in = in_dir + img_file
        img_path_out = out_dir + img_file

        image = cv2.imread(img_path_in)
        print('Image: ' + img_file + ' Shape: ' + str(image.shape))

        # classes : 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
        ones = np.ones_like(image)

        for c in range(len(CLASS_COLORS_BGR)):
            image = np.where(image==c, ones*CLASS_COLORS_BGR[c], image)

        assert np.all(image >= 0)
        assert np.all(image <= 255)

        cv2.imwrite(img_path_out, image)