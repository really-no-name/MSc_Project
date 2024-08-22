import os
import argparse
import torch
import numpy as np
from timeit import default_timer as timer

import utils.globals as globals
from utils.globals import init_global_config  # 确保导入 init_global_config
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
                model.load_state_dict(state_dict)
            else:
                model = state_dict
        except AttributeError as e:
            print(f"加载模型时发生错误: {e}")
            raise
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    return model, device

def test(model, test_data, device):
    """
    测试模型并记录结果
    """
    set_test_output_dir()
    save_image_color_legend()
    results = evaluate(model, test_data, device)
    log_results_list(results, mode='test', step=None)
    log_artifact_folder()

def evaluate(model, data, device, mode='test'):
    """
    在数据集上评估模型性能
    """
    config = globals.config
    class_no = config['data']['class_no']
    vis_images = config['data']['visualize_images'][mode]

    model.eval()
    annotator_list = data[0]
    loader_list = data[1]

    with torch.no_grad():
        results_list = []

        for e, loader in enumerate(loader_list):
            labels = []
            preds = []
            start_time = timer()
            for j, (test_img, test_label, test_name, ann_id) in enumerate(loader):
                # 将输入数据移动到同一设备
                test_img = test_img.to(device)

                if config['model']['method'] == 'pionono':
                    model.forward(test_img)
                    if 'STAPLE' in annotator_list[e] or 'MV' in annotator_list[e] or 'expert' in annotator_list[e] or config['model']['pionono_config']['always_goldpred']:
                        test_pred, _ = model.get_gold_predictions()
                    else:
                        test_pred = model.sample(use_z_mean=True, annotator_ids=ann_id, annotator_list=annotator_list)
                elif config['model']['method'] == 'prob_unet':
                    model.forward(test_img, None, training=False)
                    test_pred = model.get_gold_predictions()
                else:
                    test_pred = model(test_img)

                _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)
                test_pred_np = test_pred.cpu().detach().numpy()
                test_label = test_label.cpu().detach().numpy()
                test_label = np.argmax(test_label, axis=1)

                preds.append(test_pred_np.astype(np.int8).copy().flatten())
                labels.append(test_label.astype(np.int8).copy().flatten())

                # 我们只想对每个图像执行一次下面的代码。
                if e == 0:
                    if vis_images == 'all' or any(name in vis_images for name in test_name):
                        for k in range(len(test_name)):
                            img = test_img[k]
                            save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)
                            save_test_image_variability(model, test_name, k, mode)
            end_time = timer()
            print('Average inference time: ' + str((end_time - start_time)/len(loader_list[e])))
            preds = np.concatenate(preds, axis=0).astype(np.int8).flatten()
            labels = np.concatenate(labels, axis=0).astype(np.int8).flatten()
            if e == 0:
                shortened = False
            else:
                shortened = True
            results = get_results(preds, labels, shortened)

            print('RESULTS for ' + mode + ' Annotator: ' + str(annotator_list[e]))
            print(results)
            results_list.append(results)

    log_artifact_folder()
    return results_list


def get_results(pred, label, shortened=False):
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy().copy().flatten()
    if torch.is_tensor(label):
        label = label.cpu().detach().numpy().copy().flatten()

    results = segmentation_scores(label, pred, shortened)

    return results

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
    args = parser.parse_args()
    
    # 初始化全局配置
    init_global_config(args)
    config = globals.config

    # 设置随机种子
    torch.manual_seed(config['model']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 加载数据
    _, _, test_data, annotators = get_data()

    # 加载预训练模型
    model, device = load_pretrained_model(annotators, args.model_path, device)

    # 进行测试
    test(model, test_data, device)
