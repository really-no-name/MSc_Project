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
    start_logging()
    try:
        # load data
        trainloader, validate_data, test_data, annotators = get_data()

        # load, train and test the model
        model_handler = ModelHandler(annotators)
        model_handler.train(trainloader, validate_data)
        model_handler.test(test_data)
    except Exception as e:  # catch the error message to log it to mlflow
        f = open(os.path.join(config['logging']['experiment_folder'], 'error_message.txt'), "a")
        f.write(str(e))
        f.write(traceback.format_exc())
        f.close()
        print(e)
        print(traceback.format_exc())
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
