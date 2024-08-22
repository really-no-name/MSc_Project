import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义颜色映射
READ_COLORS_RGB = [
    [96, 255, 128],  # 类别0的颜色
    [255, 224, 32],  # 类别1的颜色
    [255, 104, 0],  # 类别2的颜色
    [255, 0, 0],  # 类别3的颜色
    [255, 255, 255]  # 类别4的颜色
]

# 创建颜色到类别的映射字典
COLOR_TO_CLASS = {tuple(color): i for i, color in enumerate(READ_COLORS_RGB)}


# 读取带有颜色的PNG掩码图像并转换为类别模式
def read_mask_image(path):
    image = Image.open(path).convert('RGB')
    array = np.array(image)
    class_array = np.zeros((array.shape[0], array.shape[1]), dtype=np.int32)
    for color, cls in COLOR_TO_CLASS.items():
        mask = np.all(array == color, axis=-1)
        class_array[mask] = cls
    return class_array, image


# 读取概率图像
def read_prob_image(path):
    image = Image.open(path).convert('L')
    array = np.array(image).astype(np.float32) / 255.0
    return array


# 将类别图像映射回颜色图像
def save_mask_image(mask, path, is_uncertainty=False):
    if is_uncertainty:
        # 将不确定性结果反转为0-255，黑色表示高不确定性
        mask = (255 * (1 - mask)).astype(np.uint8)
        img = Image.fromarray(mask, mode='L')
    else:
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls, color in enumerate(READ_COLORS_RGB):
            color_mask[mask == cls] = color
        img = Image.fromarray(color_mask)
    img.save(path)


# 平均法结合不同类别，并计算不确定性
def average_prob_with_classes(masks, prob_images, num_classes=5):
    h, w = masks[0].shape
    average_result = np.zeros((h, w), dtype=np.float32)
    uncertainty_result = np.zeros((h, w), dtype=np.float32)

    for cls in range(num_classes):
        class_masks = [(mask == cls).astype(np.float32) for mask in masks]
        class_probs = [prob_images[cls][i] for i in range(len(prob_images[cls]))]
        class_average = np.mean(class_masks, axis=0)
        class_variance = np.var(class_probs, axis=0)

        average_result[class_average > 0.5] = cls
        uncertainty_result += class_variance

    return average_result, uncertainty_result


# 显示图像
def display_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def main(name, mask_paths, prob_paths, source, show):
    # 读取并显示每个掩码图像和其数值范围
    masks = []
    for i, path in enumerate(mask_paths):
        array, image = read_mask_image(path)
        masks.append(array)
        print(f"Mask Image {i + 1} value range: {array.min()} - {array.max()}")

    # 读取每个类别的概率图像
    prob_images = {cls: [] for cls in range(5)}
    for cls in range(5):
        for i in range(len(mask_paths)):
            prob_image = read_prob_image(prob_paths[cls * len(mask_paths) + i])
            prob_images[cls].append(prob_image)

    # 使用平均法结合不同类别，并计算不确定性
    average_result, uncertainty_result = average_prob_with_classes(masks, prob_images)

    # 输出不确定性图的值的范围
    uncertainty_min = np.min(uncertainty_result)
    uncertainty_max = np.max(uncertainty_result)
    print(f"不确定性图的值范围: 最小值={uncertainty_min}, 最大值={uncertainty_max}")

    # 归一化不确定性结果到0-255范围，并反转
    uncertainty_result = (uncertainty_result - uncertainty_min) / (uncertainty_max - uncertainty_min)
    uncertainty_result = 1 - uncertainty_result  # 反转不确定性图，高不确定性为黑色

    # 保存融合结果
    output_path_average = source + '/' + name + '_AV.png'
    save_mask_image(average_result, output_path_average)

    # 保存不确定性结果
    output_path_uncertainty = source + '/' + name + '_AV_UE.png'
    save_mask_image(uncertainty_result, output_path_uncertainty, is_uncertainty=True)

    if show == 'on':
        # 显示融合结果
        display_image(average_result, 'Average with Classes Result', cmap=None)

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
        source = '/Users/Google_Drive/dataset/Gleason19/label_fusion/Train_imgs'
    elif args.mode == 'test':
        source = '/Users/Google_Drive/dataset/Gleason19/label_fusion/Test_imgs'

    mask_paths = [
        f'/Users/Google_Drive/dataset/Gleason19/output/Maps/Maps{i}_T/pred_{args.name}_mean.png' for i in range(1, 7)
    ]

    prob_paths = []
    for i in range(1, 7):
        for cls in range(0, 5):
            prob_paths.append(
                f'/Users/Google_Drive/dataset/Gleason19/output/Maps/Maps{i}_T/pred_{args.name}_class{cls}_mean_prob.png'
            )

    main(args.name, mask_paths, prob_paths, source, args.show)
