import os
import shutil
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
import SimpleITK as sitk
import multiprocessing

# 获取CPU核心数量
cpu_count = multiprocessing.cpu_count()
default_max_workers = cpu_count * 2  # 默认设置为CPU核心数的两倍

CLASS_COLORS_BGR = [
    [128, 255, 96],  # Class 0
    [32, 224, 255],  # Class 1
    [0, 104, 255],  # Class 2
    [0, 0, 255],  # Class 3
    [255, 255, 255]  # Background
]


def process_mask(mask_name, mask_dir, out_dir):
    if '._' in mask_name:
        print('Drop erroneous mask' + mask_name)
        return None
    new_name = mask_name.replace('mask_', '').replace('mask1_', '').replace('mask2_', '')
    shutil.copy(mask_dir + mask_name, out_dir + new_name)
    img_name = new_name.replace('.png', '.jpg')
    return img_name


def process_img(img_name, arv_img_dirs, out_dir, input_dir):
    for arv_img_dir in arv_img_dirs:
        arv_img_dir = input_dir + arv_img_dir
        if img_name in os.listdir(arv_img_dir):
            shutil.copy(arv_img_dir + img_name, out_dir + img_name)
            return True
    print('Image ' + img_name + ' not found!')
    return False


def process_image_resize(img_file, in_dir, out_dir, resize_res, interpolation, mask_fct=None):
    img_path_in = in_dir + img_file
    img_path_out = out_dir + img_file
    img_path_out = img_path_out.replace('.jpg', '.png').replace('_classimg_nonconvex.png', '.png')

    image = cv2.imread(img_path_in)
    image = cv2.resize(image, (resize_res, resize_res), interpolation=interpolation)
    valid = True
    if mask_fct is not None:
        image = mask_fct(image)
        assert np.all(image >= 0)
        assert np.all(image <= 4)
        if np.all(image == 4):
            valid = False
    if valid:
        cv2.imwrite(img_path_out, image)


def process_voting_image(img, config, voting_mechanism, voting_path):
    masks = []
    for a in range(len(config['map_annotator_dirs'])):
        mask_path_in = config['output_dir'] + config['map_dir'] + config['map_annotator_dirs'][a] + img
        image_array = cv2.imread(mask_path_in)
        if image_array is not None:
            masks.append(image_array)

    masks = np.array(masks)
    if masks.shape[0] > 0:
        if voting_mechanism == 'majority':
            vote_masks = stats.mode(masks, axis=0)[0][0]
        elif voting_mechanism == 'staple':
            masks_sitk_format = [sitk.GetImageFromArray(mask.astype(np.uint8)) for mask in masks]
            vote_masks_sitk_format = sitk.MultiLabelSTAPLE(masks_sitk_format)
            vote_masks = sitk.GetArrayFromImage(vote_masks_sitk_format)
            if np.any(vote_masks < 0) or np.any(vote_masks > 4) or np.any(np.mod(vote_masks, 1) != 0):
                vote_masks = stats.mode(masks, axis=0)[0][0]
        else:
            print('Choose valid voting mechanism')

        img_path_out = voting_path + img
        cv2.imwrite(img_path_out, vote_masks)
        return 1
    return 0


def convert_dataset_structure(config, max_workers=default_max_workers):
    print('### Convert dataset structure ###')
    train_img_out_dir = config['restructured_dir'] + config['train_img_dir']
    test_img_out_dir = config['restructured_dir'] + config['test_img_dir']
    ann_1_masks_out_dir = config['restructured_dir'] + config['map_dir'] + config['map_annotator_dirs'][0]
    ann_2_masks_out_dir = config['restructured_dir'] + config['map_dir'] + config['map_annotator_dirs'][1]
    os.makedirs(train_img_out_dir, exist_ok=True)
    os.makedirs(test_img_out_dir, exist_ok=True)
    os.makedirs(ann_1_masks_out_dir, exist_ok=True)
    os.makedirs(ann_2_masks_out_dir, exist_ok=True)

    print('Copy masks')

    def copy_masks_and_create_list(mask_dir, out_dir):
        masks_list = os.listdir(mask_dir)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
            results = list(
                executor.map(process_mask, masks_list, [mask_dir] * len(masks_list), [out_dir] * len(masks_list)))

        img_list = [img for img in results if img is not None]
        return img_list

    train_img_list = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_train_masks_dir'],
                                                ann_1_masks_out_dir)
    test_img_list1 = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_test_masks_dir1'],
                                                ann_1_masks_out_dir)
    test_img_list2 = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_test_masks_dir2'],
                                                ann_2_masks_out_dir)

    def check_duplicates(list, name):
        seen = set()
        dupes = [x for x in list if x in seen or seen.add(x)]
        if len(dupes) > 0:
            print('Duplicates in ' + name + ' :')
            print(dupes)
        return np.unique(list)

    train_img_list = check_duplicates(train_img_list, 'train_img_list')
    test_img_list1 = check_duplicates(test_img_list1, 'test_img_list1')
    test_img_list2 = check_duplicates(test_img_list2, 'test_img_list2')

    print('Found Train images A1: ' + str(len(train_img_list)) + ' Test images A1: ' + str(
        len(test_img_list1)) + ' Test images A2: ' + str(len(test_img_list2)))

    print('Copy images')

    def copy_list_of_imgs(img_list, out_dir):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
            executor.map(process_img, img_list, [config['arvaniti_img_dirs']] * len(img_list),
                         [out_dir] * len(img_list), [config['input_dir']] * len(img_list))

    copy_list_of_imgs(train_img_list, train_img_out_dir)
    copy_list_of_imgs(test_img_list1, test_img_out_dir)
    copy_list_of_imgs(test_img_list2, test_img_out_dir)


def process_image_resize(img_file, in_dir, out_dir, resize_res, interpolation, mask_fct=None):
    img_path_in = in_dir + img_file
    img_path_out = out_dir + img_file
    img_path_out = img_path_out.replace('.jpg', '.png').replace('_classimg_nonconvex.png', '.png')

    image = cv2.imread(img_path_in)

    # 检查图像是否正确读取
    if image is None:
        print(f"Warning: Image {img_path_in} could not be read.")
        return

    # 检查图像通道数是否为3（RGB）
    if len(image.shape) != 3 or image.shape[2] != 3:
        print(f"Warning: Image {img_path_in} does not have 3 channels.")
        return

    image = cv2.resize(image, (resize_res, resize_res), interpolation=interpolation)
    valid = True
    if mask_fct is not None:
        image = mask_fct(image)
        assert np.all(image >= 0)
        assert np.all(image <= 4)
        if np.all(image == 4):
            valid = False
    if valid:
        cv2.imwrite(img_path_out, image)


def resize_images_in_folder(config, in_dir, out_dir, resize_type='nearest', mask_fct=None,
                            max_workers=default_max_workers):
    print('### Resize ###')
    resize_res = config['resize_resolution']
    print('Processing input: ' + in_dir + ' Output: ' + out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if resize_type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif resize_type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif resize_type == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        print('Choose valid interpolation!')

    img_file_list = os.listdir(in_dir)
    print('Images found:' + str(len(img_file_list)))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
        executor.map(process_image_resize, img_file_list, [in_dir] * len(img_file_list), [out_dir] * len(img_file_list),
                     [resize_res] * len(img_file_list), [interpolation] * len(img_file_list),
                     [mask_fct] * len(img_file_list))


def resize_all_images(config, input_dir, mask_fct, max_workers=default_max_workers):
    resize_images_in_folder(config, input_dir + config['train_img_dir'], config['output_dir'] + config['train_img_dir'],
                            'bicubic', mask_fct=mask_fct, max_workers=max_workers)
    resize_images_in_folder(config, input_dir + config['test_img_dir'], config['output_dir'] + config['test_img_dir'],
                            'bicubic', mask_fct=mask_fct, max_workers=max_workers)
    for annotator_dir in config['map_annotator_dirs']:
        resize_images_in_folder(config, input_dir + config['map_dir'] + annotator_dir,
                                config['output_dir'] + config['map_dir'] + annotator_dir, 'nearest', mask_fct=mask_fct,
                                max_workers=max_workers)


def calculate_dataset_statistics(dir_path, name, list=None, file=None):
    print('--------- Dataset Statistic of ' + name, file=file)
    if list is None:
        img_file_list = os.listdir(dir_path)
    else:
        img_file_list = list

    count_pixels = [0, 0, 0, 0, 0]
    count_classes = [0, 0, 0, 0, 0]
    print('GG5 image list: ', file=file)
    for img_file in img_file_list:
        mask = cv2.imread(dir_path + img_file)
        for c in range(len(count_pixels)):
            pixels_c = int(np.sum(mask == c) / 3)
            count_pixels[c] += pixels_c
            if pixels_c > 0:
                count_classes[c] += 1
                if c == 3:
                    print("- '" + img_file + "'", file=file)

    n_all_pixels = np.sum(count_pixels)
    class_weights = n_all_pixels / (len(count_pixels) * np.array(count_pixels))
    print('Overall classes per pixel: ' + str(count_pixels), file=file)
    print('Overall classes per image: ' + str(count_classes), file=file)
    print('Class_weights: ' + str(class_weights), file=file)
    print('---------')


def create_crossvalidation_splits(config, img_dir, list_gg5, data_stat_dir='STAPLE/', max_workers=default_max_workers):
    print('### Create Cross Validation ###')
    num_splits = 4

    np.random.seed(0)
    img_file_list = os.listdir(img_dir)

    # 添加调试信息，检查路径和文件列表
    print(f"Image directory: {img_dir}")
    print(f"Image file list: {img_file_list}")

    val_n = len(img_file_list) / num_splits
    frequency_gg5 = len(img_file_list) / len(list_gg5)

    # 打印调试信息
    print(f"Image file list: {img_file_list}")
    print(f"GG5 list: {list_gg5}")

    for img in list_gg5:
        try:
            img_file_list.remove(img)
        except ValueError:
            print(f"Warning: {img} not found in image file list")

    np.random.shuffle(img_file_list)

    for i in range(len(list_gg5)):
        index = int(frequency_gg5 * i)
        img_file_list.insert(index, list_gg5[i])

    with open(config['output_dir'] + '/crossval_statistics.txt', 'w') as f:
        print('No of initial images: ' + str(len(img_file_list)))
        for i in range(num_splits):
            print('Split ' + str(i), file=f)
            val_start_id = int(val_n * i)
            val_stop_id = int(val_n * (i + 1))
            val_img_list = img_file_list[val_start_id:val_stop_id]
            crossval_dir = config['output_dir'] + 'Crossval' + str(i) + '/'
            shutil.rmtree(crossval_dir, ignore_errors=True)
            os.makedirs(crossval_dir, exist_ok=True)
            crossval_dir_train = crossval_dir + 'train/'
            os.makedirs(crossval_dir_train, exist_ok=True)
            crossval_dir_val = crossval_dir + 'val/'
            os.makedirs(crossval_dir_val, exist_ok=True)
            train_img_list = []
            for img in img_file_list:
                src_path = os.path.join(img_dir, img)
                if img in val_img_list:
                    dst_path = os.path.join(crossval_dir_val, img)
                else:
                    dst_path = os.path.join(crossval_dir_train, img)
                    train_img_list.append(img)
                try:
                    shutil.copy(src_path, dst_path)
                except FileNotFoundError as e:
                    print(f"File not found: {src_path}")

            print('No of val images: ' + str(len(os.listdir(crossval_dir_val))), file=f)
            calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + data_stat_dir, name='Validation',
                                         list=val_img_list, file=f)
            print('No of train images: ' + str(len(os.listdir(crossval_dir_train))), file=f)
            calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + data_stat_dir, name='Training',
                                         list=train_img_list, file=f)


def create_voting_masks(config, voting_mechanism='majority', dir_name='MV/', max_workers=default_max_workers):
    print('### Create Voting Maps ###')
    print('Mode: ' + voting_mechanism)
    voting_path = config['output_dir'] + config['map_dir'] + dir_name
    os.makedirs(voting_path, exist_ok=True)
    train_img_path = config['output_dir'] + config['train_img_dir']
    test_img_path = config['output_dir'] + config['test_img_dir']
    all_imgs = np.concatenate([os.listdir(train_img_path), os.listdir(test_img_path)], axis=0)
    counter = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
        results = list(
            executor.map(process_voting_image, all_imgs, [config] * len(all_imgs), [voting_mechanism] * len(all_imgs),
                         [voting_path] * len(all_imgs)))

    counter = sum(results)
    print('Total images with voting: ' + str(counter))
    calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + dir_name, voting_mechanism)


def convert_to_rgb(config, map_annotator_dirs, max_workers=default_max_workers):
    print('### Convert Maps to RGB images ###')
    rgb_dir = 'rgb_images/'
    rgb_path = config['output_dir'] + rgb_dir
    os.makedirs(rgb_path, exist_ok=True)
    map_dir = 'Maps/'

    def process_image(map_annotator_dir):
        in_dir = config['output_dir'] + map_dir + map_annotator_dir
        out_dir = rgb_path + map_annotator_dir
        os.makedirs(out_dir, exist_ok=True)
        img_file_list = os.listdir(in_dir)
        print(map_annotator_dir)
        print('Images found:' + str(len(img_file_list)))

        def convert_to_rgb(img_file):
            img_path_in = in_dir + img_file
            img_path_out = out_dir + img_file

            image = cv2.imread(img_path_in)
            ones = np.ones_like(image)

            for c in range(len(CLASS_COLORS_BGR)):
                image = np.where(image == c, ones * CLASS_COLORS_BGR[c], image)

            assert np.all(image >= 0)
            assert np.all(image <= 255)

            cv2.imwrite(img_path_out, image)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
            list(executor.map(convert_to_rgb, img_file_list))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
        executor.map(process_image, map_annotator_dirs)


def create_gold_label_proportion_folders(path, gold_dir, proportions, max_workers=default_max_workers):
    in_dir = path + gold_dir
    masks_list = os.listdir(in_dir)
    np.random.seed(0)

    def process_proportion(p):
        out_dir = path + gold_dir.replace('/', '_' + str(p) + '/')
        os.makedirs(out_dir, exist_ok=True)
        mask_selection = np.random.choice(masks_list, p, replace=False)
        for m in mask_selection:
            shutil.copy(in_dir + m, out_dir + m)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # Use max_workers parameter
        executor.map(process_proportion, proportions)
