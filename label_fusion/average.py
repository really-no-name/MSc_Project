import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 读取PNG掩码图像并转换为灰度模式
def read_mask_image(path):
    image = Image.open(path).convert('L')
    array = np.array(image)
    return array, image


# 保存结果为PNG图像
def save_mask_image(mask, path, is_uncertainty=False):
    if is_uncertainty:
        # 将不确定性结果归一化到0-255
        mask = (255 * (mask - mask.min()) / (mask.max() - mask.min())).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    img = Image.fromarray(mask)
    img.save(path)


# 平均法结合不同类别，并计算不确定性
def average_prob_with_classes(masks, num_classes=5):
    h, w = masks[0].shape
    average_result = np.zeros((h, w), dtype=np.float32)
    uncertainty_result = np.zeros((h, w), dtype=np.float32)

    for cls in range(num_classes):
        class_masks = [(mask == cls).astype(np.float32) for mask in masks]
        class_average = np.mean(class_masks, axis=0)
        class_variance = np.var(class_masks, axis=0)

        average_result[class_average > 0.5] = cls
        uncertainty_result += class_variance

    return average_result, uncertainty_result


# 显示图像
def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def main(name, mask_paths, source, show):
    # 读取并显示每个掩码图像和其数值范围
    masks = []
    for i, path in enumerate(mask_paths):
        array, image = read_mask_image(path)
        masks.append(array)
        print(f"Mask Image {i + 1} value range: {array.min()} - {array.max()}")

    # 使用平均法结合不同类别，并计算不确定性
    average_result, uncertainty_result = average_prob_with_classes(masks)

    # 输出不确定性图的值的范围
    uncertainty_min = np.min(uncertainty_result)
    uncertainty_max = np.max(uncertainty_result)
    print(f"不确定性图的值范围: 最小值={uncertainty_min}, 最大值={uncertainty_max}")

    # 保存融合结果
    output_path_average = source + '/' + name + '_AV.png'
    save_mask_image(average_result, output_path_average)

    # 保存不确定性结果
    output_path_uncertainty = source + '/' + name + '_AV_UE.png'
    save_mask_image(uncertainty_result, output_path_uncertainty, is_uncertainty=True)

    if show == 'on':
        # 显示融合结果
        display_image(average_result, 'Average with Classes Result')

        # 显示不确定性结果
        display_image(uncertainty_result, 'Uncertainty Result')

    print(output_path_average, output_path_uncertainty)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Average Prob.多类别平均法')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='Name of image')
    parser.add_argument('-show', '--show', choices=['on', 'off'], required=True,
                        help='Whether to show the image or not')

    args = parser.parse_args()

    if args.mode == 'train':
        source = 'Train_imgs'
    elif args.mode == 'test':
        source = 'Test_imgs'

    mask_paths = [
        f'/Users/Google_Drive/dataset/Gleason19/output/Maps/Maps{i}_T/pred_{args.name}_mean.png' for i in range(1, 7)
    ]

    main(args.name, mask_paths, source, args.show)
