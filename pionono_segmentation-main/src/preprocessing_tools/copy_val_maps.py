# 从Gleason19指定目录中复制每个标注者的验证掩码，这些掩码用于可视化。
# 它读取一个配置文件，确定需要复制的掩码列表，并将这些掩码从输入目录复制到输出目录


import argparse
import os
import shutil
import yaml

parser = argparse.ArgumentParser(
    description="This code copies the validation masks of each annotator that are used for visualization")
# parser.add_argument("--input_dir", "-i", type=str,
#                     default="/home/arne/datasets/Gleason_2019/resized_dataset_1024/",
#                     help="Input directory of dataset.")  # 输入路径
# parser.add_argument("--dataset_config", "-d", type=str,
#                     default="/home/arne/projects/segmentation_crowdsourcing/dev_branch/segmentation_crowdsourcing/dataset_dependent/gleason19/dataset_config_crowd_crossval0.yaml",
#                     help="Input directory of dataset config.")  # .yaml文件路径
# parser.add_argument("--output_dir", "-o", type=str,
#                     default="/home/arne/datasets/Gleason_2019/resized_dataset_1024/rgb_images/val_masks/",  # 输出路径
#                     help="Output directory of validation masks.")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/content/drive/Othercomputers/Mac/Google_Drive/dataset/Gleason19/resized_dataset_1024/",
                    help="Input directory of dataset.")  # 输入路径
parser.add_argument("--dataset_config", "-d", type=str,
                    default="/home/arne/projects/segmentation_crowdsourcing/dev_branch/segmentation_crowdsourcing/dataset_dependent/gleason19/dataset_config_crowd_crossval0.yaml",
                    help="Input directory of dataset config.")  # .yaml文件路径
parser.add_argument("--output_dir", "-o", type=str,
                    default="/content/drive/Othercomputers/Mac/Google_Drive/dataset/Gleason19/resized_dataset_1024/rgb_images/val_masks/",  # 输出路径
                    help="Output directory of validation masks.")
args = parser.parse_args()

# ------------------------------------------------------------------------
# 定义目录和标注者目录：
train_img_dir = 'Train_imgs/'
test_img_dir = 'Test_imgs/'
map_annotator_dirs = ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/', 'STAPLE', 'MV']
rgg_masks_dir = 'rgb_images/'

# ---------------------------------------------------------------------
# 读取配置文件：
# 读取YAML格式的配置文件，将其内容加载到config字典中。
with open(args.dataset_config) as file:
    config = yaml.full_load(file)


# 从配置文件中获取验证掩码的列表
masks = config['data']['visualize_images']['val']

for a in range(len(map_annotator_dirs)):
    map_annotator_dir = map_annotator_dirs[a]
    print(map_annotator_dir)
    for m in range(len(masks)):
        mask_name = masks[m]

        mask_path_in = args.input_dir + rgg_masks_dir + map_annotator_dir + mask_name
        if os.path.exists(mask_path_in):
            dir_path_out = args.output_dir + map_annotator_dir
            os.makedirs(dir_path_out, exist_ok=True)
            mask_path_out = dir_path_out + mask_name
            shutil.copy(mask_path_in, mask_path_out)

