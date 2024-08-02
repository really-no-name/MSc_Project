import argparse
import numpy as np
from scipy.stats import mode
from PIL import Image
import matplotlib.pyplot as plt

# 读取PNG掩码图像
def read_mask_image(path):
    return np.array(Image.open(path).convert('L'))

# 保存结果为PNG图像
def save_mask_image(mask, path):
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)

# 多数投票法
def majority_vote(preds):
    stacked_preds = np.stack(preds)
    majority_vote_result = mode(stacked_preds, axis=0)[0].squeeze()
    return majority_vote_result, stacked_preds

# 计算不确定性
def calculate_uncertainty(stacked_preds):
    variance = np.var(stacked_preds, axis=0)
    return variance

# 生成不确定性图并保存
def save_uncertainty_map(uncertainty, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(uncertainty, cmap='gray')
    plt.colorbar(label='Uncertainty')
    plt.title('Uncertainty Map')
    plt.savefig(path)
    # plt.show()
    plt.close()

def main(name, mask_paths, source):
    mask_paths = mask_paths
    # 读取掩码图像
    pred_masks = [read_mask_image(path) for path in mask_paths]

    # 使用多数投票法融合
    majority_result, stacked_preds = majority_vote(pred_masks)

    # 计算不确定性
    uncertainty = calculate_uncertainty(stacked_preds)

    # 输出不确定性图的值的范围
    uncertainty_min = np.min(uncertainty)
    uncertainty_max = np.max(uncertainty)
    print(f"不确定性图的值范围: 最小值={uncertainty_min}, 最大值={uncertainty_max}")

    # 保存融合结果
    output_path_majority = source + '/' + name + '_MV.png'
    save_mask_image(majority_result, output_path_majority)

    # 保存不确定性结果
    output_path_uncertainty = source + '/' + name + '_MV_UE.png'
    save_mask_image(uncertainty, output_path_uncertainty)

    # 保存不确定性图
    output_path_uncertainty_map = source + '/' + name + '_MV_UE_l.png'
    save_uncertainty_map(uncertainty, output_path_uncertainty_map)

    print("多数投票法融合结果已保存为 " + name + "_MV.png")
    print("不确定性估计结果已保存为 " + name + "_MV_UE.png")
    print("不确定性估计图已保存为 " + name + "_MV_UE_l.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Majority Vote, 多数投票法，带不确定性估计')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'], help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='文件名')

    args = parser.parse_args()

    if args.mode == 'train':
        source = 'Train_imgs'
    elif args.mode == 'test':
        source = 'Test_imgs'

    name = args.name
    # 掩码图像路径 - 请将这些路径替换为实际路径
    mask_paths = [
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps1_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps2_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps3_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps4_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps5_T/' + name + '.png',
        'dataset/Gleason19/resized_dataset_1024/Maps/Maps6_T/' + name + '.png'
    ]

    main(name, mask_paths, source)

