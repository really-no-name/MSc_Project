import os
import argparse
import traceback

import torch

import utils.globals
from data import get_data
from utils.globals import init_global_config
from model_handler import ModelHandler
from utils.mlflow_logger import start_logging, log_artifact_folder

# check training with expert labels
# check ignore last class
# check normalization


def main():
    # log metrics and artifacts with mlflow
    start_logging()  # 开始记录日志和指标
    try:
        # 加载数据
        # trainloader: 训练数据加载器，通常是一个PyTorch DataLoader对象
        # validate_data: 验证数据集
        # test_data: 测试数据集
        # annotators: 数据标注者信息或其他元数据
        trainloader, validate_data, test_data, annotators = get_data()

        # load, train and test the model
        # 初始化模型处理器
        # ModelHandler是自定义类，负责模型的加载、训练和测试
        model_handler = ModelHandler(annotators)

        # 训练模型
        # 使用训练数据(trainloader)和验证数据(validate_data)进行模型训练
        model_handler.train(trainloader, validate_data)

        # 测试模型
        # 使用测试数据(test_data)评估模型性能
        model_handler.test(test_data)

    except Exception as e:  # 捕获异常并记录错误信息
        # 打开错误日志文件，准备记录错误信息
        f = open(os.path.join(config['logging']['experiment_folder'], 'error_message.txt'), "a")

        # 将异常信息写入日志文件
        f.write(str(e))  # 记录异常的基本信息
        f.write(traceback.format_exc())  # 记录异常的详细堆栈信息
        f.close()  # 关闭文件

        print(e)
        print(traceback.format_exc())

        # 将包含错误信息的文件夹记录为 MLflow 的 artifact
        log_artifact_folder()


if __name__ == "__main__":
    print('Load configuration')
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--dataset_config", "-dc", type=str, default="./dataset_dependent/gleason19/data_configs/data_config_crossval0.yaml",
                        help="Config path (yaml file expected) to dataset config. Parameters will override defaults.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="./dataset_dependent/gleason19/experiments/cross_validation/pionono/cval0",
                        help="Config path to experiment folder. This folder is expected to contain a file called "
                             "'exp_config.yaml'. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    init_global_config(args)
    config = utils.globals.config
    torch.manual_seed(config['model']['seed'])
    main()



# import os
# import argparse
# import traceback
# import torch
# import utils.globals
# from data import get_data
# from utils.globals import init_global_config
# from model_handler import ModelHandler
# from utils.mlflow_logger import start_logging, log_artifact_folder
#
# def main():
#     start_logging()
#     try:
#         if args.predict:
#             model_handler = ModelHandler(annotators=[], predict_mode=True)
#             model_handler.predict(args.predict)
#         else:
#             # load data
#             trainloader, validate_data, test_data, annotators = get_data()
#
#             # load, train and test the model
#             model_handler = ModelHandler(annotators)
#             model_handler.train(trainloader, validate_data)
#             model_handler.test(test_data)
#     except Exception as e:
#         f = open(os.path.join(config['logging']['experiment_folder'], 'error_message.txt'), "a")
#         f.write(str(e))
#         f.write(traceback.format_exc())
#         f.close()
#         print(e)
#         print(traceback.format_exc())
#         log_artifact_folder()
#
# if __name__ == "__main__":
#     print('Load configuration')
#     parser = argparse.ArgumentParser(description="Cancer Classification")
#     parser.add_argument("--config", "-c", type=str, default="./config.yaml",
#                         help="Config path (yaml file expected) to default config.")
#     parser.add_argument("--dataset_config", "-dc", type=str, default="./dataset_dependent/gleason19/data_configs/data_config_crossval0.yaml",
#                         help="Config path (yaml file expected) to dataset config. Parameters will override defaults.")
#     parser.add_argument("--experiment_folder", "-ef", type=str, default="./dataset_dependent/gleason19/experiments/cross_validation/pionono/cval0",
#                         help="Config path to experiment folder. This folder is expected to contain a file called 'exp_config.yaml'. Parameters will override defaults. Optional.")
#     parser.add_argument("--predict", "-p", type=str, help="Path to the image for prediction.")
#     args = parser.parse_args()
#     init_global_config(args)
#     config = utils.globals.config
#     torch.manual_seed(config['model']['seed'])
#     main()
