import argparse
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

# 定义值和颜色之间的映射
value_to_class = {
    137: 1,  # NC
    192: 2,  # NC
    211: 3,  # GG4
    255: 4  # GG5
}

class_to_color = {
    0: (255, 0, 0),  # GG5
    1: (255, 104, 0),  # GG4
    2: (96, 255, 128),  # NC
    3: (255, 224, 32),  # GG3
    4: (255, 255, 255)  # bg
}


def value_to_class_mask(value_mask):
    """
    将值掩码转换为0-4的类掩码。
    """
    class_mask = np.zeros(value_mask.shape, dtype=np.uint8)
    for value, cls in value_to_class.items():
        class_mask[value_mask == value] = cls
    return class_mask


def preprocess_masks(masks):
    """
    预处理掩码，将它们分成每个类的二值掩码。
    """
    classes = np.unique(masks)
    class_masks = {cls: (masks == cls).astype(np.uint8) for cls in classes}
    return class_masks, classes


def staple(class_masks):
    """
    使用SimpleITK的STAPLE算法融合分割掩码。
    """
    fused_class_masks = {}
    uncertainties = {}
    for cls, masks in class_masks.items():
        masks_sitk = [sitk.GetImageFromArray(mask) for mask in masks]
        staple_filter = sitk.STAPLEImageFilter()
        staple_mask_sitk = staple_filter.Execute(masks_sitk)
        staple_mask = sitk.GetArrayFromImage(staple_mask_sitk)
        fused_class_masks[cls] = staple_mask

        # 计算像素级的不确定性作为掩码的标准差
        masks_array = np.array([sitk.GetArrayFromImage(mask) for mask in masks_sitk])
        uncertainty = np.std(masks_array, axis=0)
        uncertainties[cls] = uncertainty

    return fused_class_masks, uncertainties


def combine_fused_masks(fused_class_masks, classes):
    """
    将融合后的二值掩码合并为一个多类掩码。
    """
    combined_mask = np.zeros_like(next(iter(fused_class_masks.values())), dtype=np.uint8)
    for cls in classes:
        combined_mask[fused_class_masks[cls] > 0.5] = cls
    return combined_mask


def class_to_color_mask(class_mask):
    """
    将类掩码转换为颜色掩码。
    """
    color_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for cls, color in class_to_color.items():
        color_mask[class_mask == cls] = color
    return color_mask


def load_masks(mask_paths, print_state):
    """
    从指定路径加载PNG掩码，并将其转换为类掩码。
    """
    masks = []
    for path in mask_paths:
        if print_state == 'on':
            print(f"Loading mask from {path}")
        value_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if value_mask is not None:
            plt.imshow(value_mask, cmap='gray')
            plt.title(f"Original Mask: {path}")
            # plt.show()
            class_mask = value_to_class_mask(value_mask)
            plt.imshow(class_mask, cmap='gray')
            plt.title(f"Class Mask: {path}")
            # plt.show()
            masks.append(class_mask)
            if print_state == 'on':
                print(f"Mask shape: {class_mask.shape}, unique values: {np.unique(class_mask)}")
        else:
            print(f"Warning: Mask at {path} is None or could not be loaded.")
    if not masks:
        raise ValueError("No masks were loaded. Please check the input paths.")
    return np.array(masks)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def save_fused_mask(fused_mask, output_path, print_state):
    """
    将融合后的类掩码转换为颜色掩码并保存到指定路径。
    """
    color_mask = class_to_color_mask(fused_mask)
    plt.imshow(color_mask)
    plt.title("Fused Mask with Colors")
    # plt.show()

    # Convert RGB to BGR before saving
    bgr_color_mask = rgb_to_bgr(color_mask)
    cv2.imwrite(output_path, bgr_color_mask)
    if print_state == 'on':
        print(f"Fused mask saved to {output_path}")


def main(mask_paths, fused_output_path, print_state):
    if print_state == 'on':
        print("Starting the mask fusion process...")
    masks = load_masks(mask_paths, print_state)
    if print_state == 'on':
        print(f"Loaded {len(masks)} masks.")

    class_masks, classes = preprocess_masks(masks)
    if print_state == 'on':
        print(f"Classes found: {classes}")

    fused_class_masks, uncertainties = staple(class_masks)

    fused_mask = combine_fused_masks(fused_class_masks, classes)
    plt.imshow(fused_mask, cmap='gray')
    plt.title("Fused Class Mask")
    # plt.show()

    if print_state == 'on':
        print(f"Fused mask unique values: {np.unique(fused_mask)}")

    if np.all(fused_mask == 0):
        print("Warning: The fused mask is completely black.")

    if print_state == 'on':
        print("Mask fusion completed.")
    save_fused_mask(fused_mask, fused_output_path, print_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Staple 模型投票，0-4分类')
    parser.add_argument('-p', '--print_state', type=str, default='on', choices=['on', 'off'], help='是否print')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name to use for image paths')
    args = parser.parse_args()

    if args.mode == 'train':
        source = 'dataset/Gleason19/label_fusion/Train_imgs'
    elif args.mode == 'test':
        source = 'dataset/Gleason19/label_fusion/Test_imgs'

    name = args.name

    print('------------------------------  ' + name + 'Label Fusion ------------------------')

    mask_paths = [
        'dataset/Gleason19/output/Maps/Maps1_T/pred_' + name + '_mean.png',
        'dataset/Gleason19/output/Maps/Maps2_T/pred_' + name + '_mean.png',
        'dataset/Gleason19/output/Maps/Maps3_T/pred_' + name + '_mean.png',
        'dataset/Gleason19/output/Maps/Maps4_T/pred_' + name + '_mean.png',
        'dataset/Gleason19/output/Maps/Maps5_T/pred_' + name + '_mean.png',
        'dataset/Gleason19/output/Maps/Maps6_T/pred_' + name + '_mean.png'
    ]

    fused_output_path = source + '/' + name + '_STAPLEcolor.png'

    main(mask_paths, fused_output_path, args.print_state)
