import argparse
import time
from datetime import datetime

import numpy as np
from scipy.stats import mode
from PIL import Image
import matplotlib.pyplot as plt

# 读取PNG掩码图像并转换为类别标签
def read_mask_image(path, class_colors_rgb):
    mask = np.array(Image.open(path))
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for label, color in enumerate(class_colors_rgb):
        matches = np.all(mask == color, axis=-1)
        label_mask[matches] = label
        # print(f"颜色 {color} 匹配的像素数量: {np.sum(matches)}")
    # print(f"读取的图像路径: {path}")
    # print(f"读取的图像唯一值: {np.unique(mask.reshape(-1, mask.shape[2]), axis=0)}")
    # print(f"转换后的标签掩码唯一值: {np.unique(label_mask)}")
    return label_mask

# 读取PNG概率图像
def read_prob_image(path):
    return np.array(Image.open(path).convert('L')) / 255.0

# 保存结果为彩色PNG图像
def save_colored_mask_image(mask, path, class_colors_rgb):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(class_colors_rgb):
        color_mask[mask == label] = color
    img = Image.fromarray(color_mask)
    img.save(path)
    # print(f"保存的彩色掩码图像路径: {path}")
    # print(f"保存的彩色掩码图像唯一值: {np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)}")

# 保存结果为灰度PNG图像
def save_grayscale_image(image, path):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(path)
    # print(f"保存的灰度图像路径: {path}")
    # print(f"保存的灰度图像唯一值: {np.unique(image)}")

# 多数投票法
def majority_vote(preds):
    stacked_preds = np.stack(preds)
    majority_vote_result = mode(stacked_preds, axis=0)[0].squeeze()
    # print(f"多数投票结果唯一值: {np.unique(majority_vote_result)}")
    return majority_vote_result, stacked_preds

# 计算不确定性
def calculate_uncertainty(prob_maps):
    uncertainty = np.var(prob_maps, axis=0)
    # print(f"计算的不确定性唯一值: {np.unique(uncertainty)}")
    return uncertainty

# 生成不确定性图并保存
def save_uncertainty_map(uncertainty, path):
    normalized_uncertainty = (uncertainty / np.max(uncertainty) * 255).astype(np.uint8)
    # print(f"归一化后不确定性图唯一值: {np.unique(normalized_uncertainty)}")
    img = Image.fromarray(normalized_uncertainty)
    img.save(path)
    # print(f"保存的不确定性图路径: {path}")

def main(name, mask_paths, prob_paths, source, read_colors_rgb, save_colors_rgb):
    mask_paths = mask_paths
    prob_paths = prob_paths

    # 读取掩码图像
    pred_masks = [read_mask_image(path, read_colors_rgb) for path in mask_paths]

    # 使用多数投票法融合
    majority_result, stacked_preds = majority_vote(pred_masks)

    # 读取概率图像
    prob_maps = [read_prob_image(path) for path in prob_paths]

    # 计算不确定性
    uncertainty = calculate_uncertainty(prob_maps)

    # 输出不确定性图的值的范围
    uncertainty_min = np.min(uncertainty)
    uncertainty_max = np.max(uncertainty)
    # print(f"不确定性图的值范围: 最小值={uncertainty_min}, 最大值={uncertainty_max}")

    # 保存融合结果
    # output_path_majority = source + '/' + name + '_MV.png'
    # save_colored_mask_image(majority_result, output_path_majority, save_colors_rgb)

    # 保存不确定性结果
    # output_path_uncertainty = source + '/' + name + '_MV_UE.png'
    # save_grayscale_image(uncertainty, output_path_uncertainty)

    # 保存不确定性图
    output_path_uncertainty_map = source + '/' + name + '_MV_UE.png'
    save_uncertainty_map(uncertainty, output_path_uncertainty_map)

    # print("多数投票法融合结果已保存为 " + name + "_MV.png")
    print("不确定性估计结果已保存为 " + name + "_MV_UE.png")
    # print("不确定性估计图已保存为 " + name + "_MV_UE_l.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Majority Vote, 多数投票法，只生成不确定性图（无标签）')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'], help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='文件名')

    args = parser.parse_args()

    if args.mode == 'train':
        source = 'dataset/Arvaniti_TMA/label_fusion/Train_imgs'
    elif args.mode == 'test':
        source = 'dataset/Arvaniti_TMA/label_fusion/Test_imgs'

    name = args.name
    # 掩码图像路径 - 请将这些路径替换为实际路径
    mask_paths = [
        f'dataset/Arvaniti_TMA/output/Maps/Maps{i}_T/pred_{name}_mean.png' for i in range(1, 7)
    ]

    # 概率图像路径
    prob_paths = [
        f'dataset/Arvaniti_TMA/output/Maps/Maps{i}_T/pred_{name}_class{cls}_mean_prob.png'
        for i in range(1, 7) for cls in range(0, 5)
    ]

    # 定义RGB颜色映射
    READ_COLORS_RGB = [
        [96, 255, 128],   # 类别0的颜色
        [255, 224, 32],   # 类别1的颜色
        [255, 104, 0],    # 类别2的颜色
        [255, 0, 0],      # 类别3的颜色
        [255, 255, 255]   # 类别4的颜色
    ]

    SAVE_COLORS_RGB = READ_COLORS_RGB

    start_time = time.time()
    main(name, mask_paths, prob_paths, source, READ_COLORS_RGB, SAVE_COLORS_RGB)
    end_time = time.time()
    print(f"Elapsed time for img {args.name}: {end_time - start_time:.6f} seconds")
    # 获取当前时间
    current_time = datetime.now()

    # 输出当前时间
    print("结束时间:", current_time)