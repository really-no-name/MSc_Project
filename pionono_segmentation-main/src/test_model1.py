import os
import argparse
import torch
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from torchvision import transforms
import utils.globals as globals
from utils.globals import init_global_config
from utils.saving import save_test_images, save_image_color_legend, save_test_image_variability
from utils.test_helpers import segmentation_scores
from utils.mlflow_logger import log_results_list, log_artifact_folder, set_test_output_dir
from utils.initialize_model import init_model
from data import get_data

def load_pretrained_model(annotators, model_path, device):
    """
    加载预训练模型
    """
    model = init_model(annotators)
    model.to(device)  # 将模型移动到指定设备
    
    # 确保模型文件存在，然后加载模型权重
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            print(f"加载的 state_dict 类型: {type(state_dict)}")
            if isinstance(state_dict, dict):
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"缺少的键: {missing_keys}")
                print(f"未预料到的键: {unexpected_keys}")
            else:
                model = state_dict
        except AttributeError as e:
            print(f"加载模型时发生错误: {e}")
            raise
        except ModuleNotFoundError as e:
            print(f"模块未找到错误: {e}")
            raise
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    return model, device

def load_unlabeled_data(img_dir, device):
    """
    加载未标记的图像数据
    """
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])
    
    img_files = sorted(os.listdir(img_dir))  # 获取图像文件列表
    images = []
    image_names = []

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')  # 确保图像是 RGB 格式
        img = transform(img).unsqueeze(0)  # 增加批次维度
        images.append(img.to(device))  # 将图像移动到设备上
        image_names.append(img_file)

    # 检查加载的图像数量和名称
    print(f"加载了 {len(images)} 张图像")
    print("图像名称示例:", image_names[:5])

    return images, image_names

def predict_and_save_maps(model, images, image_names, output_dir, device):
    """
    生成分割图像并保存
    """
    model.eval()  # 设置模型为评估模式

    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    with torch.no_grad():
        for i, img in enumerate(images):
            # 打印输入图像的形状
            print(f"输入图像 {i} 的形状: {img.shape}")

            # 进行推理
            output = model(img)

            # 检查模型输出
            if output is None:
                print(f"模型输出为 None，检查模型推理。图像索引: {i}")
                continue

            # 打印输出张量的形状
            print(f"输出张量的形状: {output.shape}")

            # 获取模型输出，选择最可能的类别
            pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # 将分割结果保存为图像
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_names[i])[0]}_map.png")
            pred_img = Image.fromarray((pred_map * 255).astype(np.uint8))  # 转换为 0-255 的图像
            pred_img.save(save_path)
            print(f"保存分割图像: {save_path}")

if __name__ == "__main__":
    print('加载配置文件')
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--dataset_config", "-dc", type=str, default="./dataset_dependent/gleason19/data_configs/data_config_crossval0.yaml",
                        help="Config path (yaml file expected) to dataset config. Parameters will override defaults.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="./dataset_dependent/gleason19/experiments/cross_validation/pionono/cval0",
                        help="Config path to experiment folder. This folder is expected to contain a file called "
                             "'exp_config.yaml'. Parameters will override defaults. Optional.")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to the pre-trained model file.")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Path to the directory containing unlabeled test images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the directory to save the generated maps.")
    args = parser.parse_args()
    
    # 初始化全局配置
    init_global_config(args)
    config = globals.config

    # 设置随机种子
    torch.manual_seed(config['model']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    images, image_names = load_unlabeled_data(args.img_dir, device)

    # 加载预训练模型
    _, _, _, annotators = get_data()  # 获取 annotators

    model, device = load_pretrained_model(annotators, args.model_path, device)

    # 使用模型生成分割图像并保存
    predict_and_save_maps(model, images, image_names, args.output_dir, device)
