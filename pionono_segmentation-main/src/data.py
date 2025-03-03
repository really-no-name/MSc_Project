import os
import torch
import numpy as np
import pandas as pd
import cv2
import random

from torch.utils import data

import albumentations as albu
import utils.globals as globals
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.preprocessing import get_preprocessing_fn_without_normalization


def get_training_augmentation(ignore_class=4):
    """
    获取训练数据的增强（augmentation）配置。

    参数:
        ignore_class (int): 在图像增强过程中，用于填充掩码（mask）的忽略类别值。默认值为4。

    返回:
        composed_transform (albu.Compose or None): 如果启用数据增强，则返回一个组合的增强变换对象；
                                                      否则返回None。
    """
    # 从全局配置中获取数据增强的配置
    aug_config = globals.config['data']['augmentation']

    # 检查是否启用了数据增强
    if aug_config['use_augmentation']:
        train_transform = [  # 定义训练数据的增强变换列表
            albu.HorizontalFlip(p=0.5),  # 水平翻转，概率为50%
            albu.VerticalFlip(p=0.5),  # 垂直翻转，概率为50%
            albu.Blur(blur_limit=aug_config['gaussian_blur_kernel'], p=0.8),  # 高斯模糊，模糊核大小由配置指定，概率为80%

            # 随机调整亮度和对比度，亮度和对比度的限制由配置指定，概率为100%
            albu.RandomBrightnessContrast(brightness_limit=aug_config['brightness_limit'],
                                          contrast_limit=aug_config['contrast_limit'],
                                          p=1.0),

            # 随机调整色调、饱和度和值（HSV），限制由配置指定，概率为100%
            albu.HueSaturationValue(hue_shift_limit=aug_config['hue_shift_limit'],
                                    sat_shift_limit=aug_config['sat_shift_limit'],
                                    p=1.0),

            # 仿射变换，包括缩放、平移、剪切和旋转，插值方法为双三次插值，
            # 填充颜色为白色（RGB值为[255, 255, 255]），掩码填充值为ignore_class，概率为100%
            albu.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), shear=[-5, 5],
                        rotate=[-360, 360], interpolation=cv2.INTER_CUBIC, cval=[255, 255, 255], cval_mask=ignore_class, p=1.0)
        ]

        # 将增强变换组合成一个Compose对象
        composed_transform = albu.Compose(train_transform)
    else:
        composed_transform = None  # 如果未启用数据增强，则返回None
    return composed_transform  # 返回组合的增强变换对象或None


def to_tensor(x, **kwargs):
    """
    将输入的numpy数组转换为PyTorch张量。

    参数:
        x (numpy.ndarray): 输入的numpy数组，通常是一个图像或特征矩阵。
        **kwargs: 额外的关键字参数（未使用，仅为兼容性保留）。

    返回:
        numpy.ndarray: 转换后的数组，形状为(C, H, W)，数据类型为float32。
    """
    # 将数组的维度从(H, W, C)转换为(C, H, W)
    # PyTorch的输入格式通常为通道优先（Channel-first）
    # x = x.transpose(2, 0, 1)

    # 将数据类型转换为float32，以便与PyTorch兼容
    # x = x.astype('float32')

    # 返回转换后的数组
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    构造数据预处理的变换管道。

    参数:
        preprocessing_fn (callable): 数据归一化函数，通常是一个针对特定预训练神经网络的归一化函数。
                                        例如，对输入图像进行标准化（如减去均值并除以标准差）。

    返回:
        transform (albu.Compose): 一个包含预处理步骤的albumentations.Compose对象。
    """
    # 定义预处理变换列表
    _transform = [
        albu.Lambda(image=preprocessing_fn),  # 对图像应用归一化函数
        albu.Lambda(image=to_tensor, mask=to_tensor),  # 将图像和掩码（如果有）转换为PyTorch张量格式
    ]
    return albu.Compose(_transform)  # 将变换列表组合成一个Compose对象

# =============================================

# class SupervisedDataset(torch.utils.data.Dataset):
#     """Supervised Dataset. Read images, apply augmentation and preprocessing transformations.
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing
#             (e.g. normalization, shape manipulation, etc.)
#     """
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             augmentation=None,
#             preprocessing=None
#     ):
#         img_ids = os.listdir(images_dir)
#         mask_ids = os.listdir(masks_dir)
#         self.ids = np.intersect1d(img_ids, mask_ids)
#         if self.ids.size == 0:
#             raise Exception('Empty data generator because no images with masks were found.')
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#         self.class_no = globals.config['data']['class_no']
#         self.class_values = self.set_class_values(self.class_no)
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#
#     def __getitem__(self, i):
#
#         # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], 0)
#         if mask is None:
#             raise Exception('Empty mask! Path: ' + self.masks_fps[i])
#
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype('float')
#
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#         return image, mask, self.ids[i], 0
#
#     def __len__(self):
#         return len(self.ids)
#
#     def set_class_values(self, class_no):
#         if globals.config['data']['ignore_last_class']:
#             class_values = list(range(class_no + 1))
#         else:
#             class_values = list(range(class_no))
#         return class_values


class Dataset(torch.utils.data.Dataset):
    """Crowdsourced_Dataset Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        image_path (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            data_path,
            image_path,
            masks_dirs,
            augmentation=None,
            preprocessing=None,
            repeat_images=None,
            repeat_factor=1,
            annotator_ids='auto',
            _set=None
    ):
        """
            初始化数据集类。

            参数:
                data_path (str): 数据集的根目录路径。
                image_path (str): 图像文件夹的相对路径（相对于data_path）。
                masks_dirs (list): 掩码文件夹的相对路径列表（相对于data_path）。
                augmentation (albu.Compose, optional): 数据增强的变换管道。默认为None。
                preprocessing (albu.Compose, optional): 数据预处理的变换管道。默认为None。
                repeat_images (list, optional): 需要重复的图像ID列表。默认为None。
                repeat_factor (int, optional): 重复图像的次数。默认为1。
                annotator_ids (str or list, optional): 标注者ID列表或设置为'auto'。默认为'auto'。
                _set (str, optional): 数据集类型（如'train', 'val', 'test'）。默认为None。
            """
        # 拼接图像文件夹的完整路径
        image_path = os.path.join(data_path, image_path)

        # 拼接掩码文件夹的完整路径
        mask_paths = [os.path.join(data_path, m) for m in masks_dirs]

        # 提取标注者名称（掩码文件夹的名称）
        self.annotators = [x.split('/')[-1] for x in masks_dirs]
        self.mask_paths = mask_paths

        # 获取有效的图像ID列表（确保图像和掩码文件都存在）
        self.ids = self.get_valid_ids(os.listdir(image_path), mask_paths, repeat_images, repeat_factor)

        # 拼接图像的完整路径
        self.images_fps = [os.path.join(image_path, image_id) for image_id in self.ids]

        # 设置标注者数量和类别数量
        self.annotators_no = len(self.annotators)
        self.class_no = globals.config['data']['class_no']

        # 设置类别值（通常为0到class_no-1）
        self.class_values = self.set_class_values(self.class_no)

        # 设置数据增强和预处理
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # 设置标注者ID
        self.annotator_ids = annotator_ids

        # 设置忽略的类别索引
        if globals.config['data']['ignore_last_class']:
            # 如果忽略最后一个类别，则将其索引设置为class_no
            self.ignore_index = int(self.class_no)  # deleted class is always set to the last index
        else:
            # 否则设置为-100，表示不忽略任何类别
            self.ignore_index = -100  # this means no index ignored

    def __getitem__(self, i):
        """
        获取数据集中第i个样本。

        参数:
            i (int): 样本的索引。

        返回:
            如果存在掩码数据:
                image (numpy.ndarray): 预处理后的图像数据。
                mask (numpy.ndarray): 预处理后的掩码数据。
                image_id (str): 图像的ID。
                annotator_id (int): 标注者的ID。
            如果不存在掩码数据:
                image (numpy.ndarray): 预处理后的图像数据。
                image_id (str): 图像的ID。
        """
        # read data
        image = cv2.imread(self.images_fps[i])  # 读取图像文件
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式

        # 检查是否存在标注数据
        if self.mask_paths:
            mask_found = False
            indexes = np.random.permutation(self.annotators_no)  # 随机打乱标注者顺序，以便随机选择标注者
            for ann_index in indexes:
                ann_path = self.mask_paths[ann_index]  # 获取当前标注者的掩码路径
                mask_path = os.path.join(ann_path, self.ids[i])  # 拼接掩码文件的完整路径
                if os.path.exists(mask_path):  # 检查掩码文件是否存在
                    mask = cv2.imread(mask_path, 0)  # 读取掩码文件（灰度模式）
                    id = self.mask_paths.index(ann_path)  # 获取当前标注者的索引
                    if self.annotator_ids == 'auto':
                        annotator_id = id  # 如果annotator_ids为'auto'，则使用索引作为标注者ID
                    else:
                        annotator_id = self.annotator_ids[id]  # 否则使用指定的标注者ID
                    mask_found = True
                    break  # 找到掩码后退出循环

            if not mask_found:  # 如果没有找到掩码，抛出异常
                raise Exception('No mask was found for image: ' + self.images_fps[i])

            # apply augmentations
            if self.augmentation:  # 应用数据增强
                sample = self.augmentation(image=image, mask=mask)  # 对图像和掩码应用增强
                image = sample['image']
                mask = sample['mask']

            # 将掩码转换为多通道格式（每个类别一个通道）
            mask = [(mask == v) for v in self.class_values]  # 为每个类别生成一个二进制掩码
            mask = np.stack(mask, axis=-1).astype('float')  # 将掩码堆叠为多通道数组

            # apply preprocessing  # 应用数据预处理
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)  # 对图像和掩码应用预处理
                image = sample['image']
                mask = sample['mask']

            return image, mask, self.ids[i], annotator_id  # 返回图像、掩码、图像ID和标注者ID
        else:
            # apply preprocessing without mask  # 如果没有掩码数据，仅对图像应用预处理
            if self.preprocessing:
                image = self.preprocessing(image=image)['image']
            return image, self.ids[i]  # 返回图像和图像ID

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        """
        设置类别值列表。

        参数:
            class_no (int): 类别的数量。

        返回:
            class_values (list): 类别值的列表。
        """
        # 检查是否忽略最后一个类别
        if globals.config['data']['ignore_last_class']:
            # 如果忽略最后一个类别，则类别值包括0到class_no（含class_no）
            class_values = list(range(class_no + 1))
        else:
            # 否则，类别值包括0到class_no-1
            class_values = list(range(class_no))
        return class_values  # 返回类别值列表

    def get_valid_ids(self, image_ids, mask_paths, repeat_images=None, repeat_factor=0):
        """
        Returns all image ids that have at least one corresponding annotated mask
        返回所有至少有一个对应掩码的图像ID。

        参数:
            image_ids (list): 所有图像文件的ID列表。
            mask_paths (list): 所有掩码文件夹的路径列表。
            repeat_images (list, optional): 需要重复的图像ID列表。默认为None。
            repeat_factor (int, optional): 重复图像的次数。默认为0。

        返回:
            valid_ids (numpy.ndarray): 有效的图像ID数组。
        """
        all_masks = []  # 收集所有掩码文件的ID
        for p in range(len(mask_paths)):
            mask_ids = os.listdir(mask_paths[p])  # 获取当前掩码文件夹中的所有文件ID
            for m in mask_ids:
                all_masks.append(m)  # 将掩码文件ID添加到列表中

        all_unique_masks = np.unique(all_masks)  # 获取所有唯一的掩码文件ID
        valid_ids = np.intersect1d(image_ids, all_unique_masks)  # 获取图像ID和掩码文件ID的交集，即有效的图像ID

        # 如果需要重复某些图像ID
        if repeat_images is not None:
            # 获取需要重复的图像ID（必须是有效的图像ID）
            repeat_ids = np.intersect1d(valid_ids, repeat_images)
            # 根据重复次数将需要重复的图像ID添加到有效图像ID列表中
            for i in range(repeat_factor):
                valid_ids = np.concatenate([valid_ids, repeat_ids], axis=0)

        return valid_ids  # 返回有效的图像ID数组


def get_data(load_train=False, load_val=False):
    """
    加载训练、验证和测试数据集。

    参数:
        load_train (bool): 是否加载训练数据集。默认为False。
        load_val (bool): 是否加载验证数据集。默认为False。

    返回:
        trainloader (torch.utils.data.DataLoader): 训练数据加载器。
        validate_data (tuple): 验证数据集信息，包含掩码路径和数据加载器列表。
        test_data (tuple): 测试数据集信息，包含掩码路径和数据加载器列表。
        annotators (list): 标注者列表。
    """
    config = globals.config  # 获取全局配置
    batch_size = config['model']['batch_size']  # 从配置中获取批量大小
    normalization = config['data']['normalization']  # 从配置中获取是否进行归一化
    class_no = config['data']['class_no'] - 1 + int(config['data']['ignore_last_class'])  # 计算类别数量（如果忽略最后一个类别，则类别数量减1）

    # 根据是否归一化选择预处理函数
    if normalization:
        encoder_name = config['model']['encoder']['backbone']  # 编码器名称
        encoder_weights = config['model']['encoder']['weights']  # 编码器权重
        # 获取归一化预处理函数
        preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
    else:
        preprocessing_fn = get_preprocessing_fn_without_normalization()  # 获取无归一化的预处理函数

    # 获取预处理管道
    preprocessing = get_preprocessing(preprocessing_fn)

    # 初始化训练数据加载器、验证数据加载器和标注者列表
    trainloader = None
    validateloaders = []
    annotators = []

    # 加载训练数据集
    if load_train:
        # 加载训练数据集
        train_dataset = Dataset(
            config['data']['path'],  # 数据集根路径
            config['data']['train']['images'],  # 训练图像路径
            config['data']['train']['masks'],  # 训练掩码路径
            augmentation=get_training_augmentation(ignore_class=class_no),   # 数据增强
            preprocessing=preprocessing,  # 数据预处理
            repeat_images=config['data']['repeat_train_images'],  # 需要重复的图像
            repeat_factor=config['data']['repeat_factor']  # 重复次数
        )
        annotators = train_dataset.annotators  # 获取标注者列表

        # 创建训练数据加载器
        trainloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,  # 批量大小
            shuffle=True,  # 是否打乱数据
            num_workers=8,  # 数据加载的线程数
            drop_last=True  # 是否丢弃最后一个不完整的批次
        )

    # 加载验证数据集
    if load_val:
        # 验证：为每个标注者创建单独的加载器
        val_masks = config['data']['val']['masks']  # 验证掩码路径
        for a in range(len(val_masks)):
            # 为每个标注者创建验证数据集
            validate_dataset = Dataset(
                config['data']['path'],  # 数据集根路径
                config['data']['val']['images'],  # 验证图像路径
                [val_masks[a]],  # 当前标注者的掩码路径
                preprocessing=preprocessing,  # 数据预处理
                annotator_ids=[a]  # 标注者ID
            )
            # 创建验证数据加载器
            validateloaders.append(data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False))
        validate_data = (val_masks, validateloaders)  # 验证数据集信息
    else:
        validate_data = ([], [])  # 如果不加载验证数据，返回空列表

    # 测试：为每个标注者创建单独的加载器
    test_masks = config['data']['test'].get('masks', [])  # 如果没有提供标注，默认为空列表
    test_images_path = config['data']['test']['images']  # 测试图像路径

    if test_masks:
        testloaders = []
        for a in range(len(test_masks)):
            # 为每个标注者创建测试数据集
            test_dataset = Dataset(
                config['data']['path'],  # 数据集根路径
                test_images_path,  # 测试图像路径
                [test_masks[a]],  # 当前标注者的掩码路径
                preprocessing=preprocessing,  # 数据预处理
                annotator_ids=[a]  # 标注者ID
            )

            # 创建测试数据加载器
            testloaders.append(data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False))
        test_data = (test_masks, testloaders)  # 测试数据集信息
    else:
        # 如果没有标注数据，仅加载图像
        test_dataset = Dataset(
            config['data']['path'],  # 数据集根路径
            test_images_path,  # 测试图像路径
            [],  # 无掩码路径
            preprocessing=preprocessing,  # 数据预处理
            annotator_ids=[]  # 无标注者ID
        )

        # 测试数据集信息
        test_data = ([], [data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)])

    # 返回训练数据加载器、验证数据集信息、测试数据集信息和标注者列表
    return trainloader, validate_data, test_data, annotators



