import os
import imageio
import errno
import csv
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import utils.globals as globals
import matplotlib.colors as mcolors
import torch.nn.functional as F

# for classes: NC, GG3, GG4, GG5, background
CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]

def save_model(model):
    dir = globals.config['logging']['experiment_folder']
    out_path = os.path.join(dir, 'model.pth')
    torch.save(model, out_path)
    print('Model saved to: ' + out_path)


def save_test_images(test_imgs:torch.Tensor, test_preds: np.array, test_labels: np.array, test_name: np.array, mode: str):
    visual_dir = 'qualitative_results/' + mode
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    # 处理无标注数据的情况
    if test_labels is not None:
        if len(test_labels.shape) == 3:
            test_labels = test_labels[0]

        test_preds = np.asarray(test_preds, dtype=np.uint8)
        test_labels = np.asarray(test_labels, dtype=np.uint8)

        # print("test name ", test_name)
        out_path = os.path.join(dir, 'img_' + test_name)
        save_image(test_imgs, out_path)

        test_pred_rgb = convert_classes_to_rgb(test_preds)
        out_path = os.path.join(dir, 'pred_' + test_name)
        imageio.imsave(out_path, test_pred_rgb)

        test_label_rgb = convert_classes_to_rgb(test_labels)
        out_path = os.path.join(dir, 'gt_' + test_name)
        imageio.imsave(out_path, test_label_rgb)
    else:
        # 仅保存预测结果和原图
        out_path = os.path.join(dir, 'img_' + test_name)
        save_image(test_imgs, out_path)

        test_pred_rgb = convert_classes_to_rgb(test_preds)
        out_path = os.path.join(dir, 'pred_' + test_name)
        imageio.imsave(out_path, test_pred_rgb)



def save_test_image_variability(model, test_name, k, mode):
    no_samples_per_annotator = 12
    annotators = globals.config['data']['val']['masks']
    method = globals.config['model']['method']
    class_no = globals.config['data']['class_no']
    visual_dir = 'qualitative_results/' + mode
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    dir = os.path.join(dir, 'variability')
    os.makedirs(dir, exist_ok=True)
    if method == 'pionono' and not globals.config['model']['pionono_config']['always_goldpred']:
        for i in range(len(annotators)):
            a = annotators[i]
            a_dir = os.path.join(dir, a)
            os.makedirs(a_dir, exist_ok=True)
            if ('STAPLE' in annotators[i] or 'MV' in annotators[i]):
                pred, std = model.get_gold_predictions()
                probabilities = F.softmax(pred[:, 0:class_no], dim=1)  # 获取预测概率

                # 调试信息
                # print(f"Probabilities shape: {probabilities.shape}")
                # print(
                #     f"Probabilities min: {probabilities.min().item()}, max: {probabilities.max().item()}, mean: {probabilities.mean().item()}")

                _, pred = torch.max(pred[:, 0:class_no], dim=1)
                out_path = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_gold" + ".png"))
                pred_k = convert_classes_to_rgb(pred[k].cpu().detach().numpy())
                imageio.imsave(out_path, pred_k)
                # print('保存gold图: ' + 'pred_' + test_name[k].replace(".png", "_gold" + ".png"))

                std = torch.mean(std[:, 0:class_no], dim=1)
                out_path_std = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_gold_var" + ".png"))
                var_k = convert_std_to_rgb(std[k].cpu().detach().numpy())
                imageio.imsave(out_path_std, var_k)
                # print('保存gold_var图: ' + 'pred_' + test_name[k].replace(".png", "_gold_var" + ".png"))

                # 保存概率图像 XU
                prob_k = convert_probabilities_to_rgb(
                    probabilities[0].cpu().detach().numpy().transpose(1, 2, 0), 'type1', a_dir, test_name[k], 1)  # 修改索引
                out_path_prob = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_prob" + ".png"))
                imageio.imsave(out_path_prob, prob_k)
                # print('存储概率图：', 'pred_' + test_name[k].replace(".png", "_prob" + ".png"))

            else:
                annotator = torch.ones(model.unet_features.shape[0]) * i
                mean_pred = model.sample(use_z_mean=True, annotator_ids=annotator, annotator_list=annotators)
                probabilities = F.softmax(mean_pred[:, 0:class_no], dim=1)  # 获取预测概率

                # 调试信息
                # print(f"Mean probabilities shape: {probabilities.shape}")
                # print(
                #     f"Mean probabilities min: {probabilities.min().item()}, max: {probabilities.max().item()}, mean: {probabilities.mean().item()}")

                _, mean_pred = torch.max(mean_pred[:, 0:class_no], dim=1)
                mean_pred_k = convert_classes_to_rgb(mean_pred[k].cpu().detach().numpy())
                out_path = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_mean" + ".png"))
                imageio.imsave(out_path, mean_pred_k)
                # print('存储mean图：', 'pred_' + test_name[k].replace(".png", "_mean" + ".png"))

                # 保存概率图像 XU
                prob_k = convert_probabilities_to_rgb(
                    probabilities[0].cpu().detach().numpy().transpose(1, 2, 0), 'type2', a_dir, test_name[k], 1)  # 修改索引
                # out_path_prob = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_mean_prob" + ".png"))
                # imageio.imsave(out_path_prob, prob_k)
                # print('存储概率图：', 'pred_' + test_name[k].replace(".png", "_mean_prob" + ".png"))

                for s in range(no_samples_per_annotator -1):
                    pred = model.sample(use_z_mean=False, annotator_ids=annotator, annotator_list=annotators)
                    # probabilities = F.softmax(pred[:, 0:class_no], dim=1)  # 获取预测概率

                    # 调试信息
                    # print(f"Sample {s} probabilities shape: {probabilities.shape}")
                    # print(
                    #     f"Sample {s} probabilities min: {probabilities.min().item()}, max: {probabilities.max().item()}, mean: {probabilities.mean().item()}")

                    _, pred = torch.max(pred[:, 0:class_no], dim=1)
                    out_path = os.path.join(a_dir, 'pred_' + test_name[k].replace(".png", "_s_" + str(s) + ".png"))
                    pred_k = convert_classes_to_rgb(pred[k].cpu().detach().numpy())
                    imageio.imsave(out_path, pred_k)
                    # print('存储多样性采样图：', 'pred_' + test_name[k].replace(".png", "_s_" + str(s) + ".png"))

                    # 保存概率图像 XU
                    # prob_k = convert_probabilities_to_rgb(
                    #     probabilities[0].cpu().detach().numpy().transpose(1, 2, 0), 'type3', a_dir, test_name[k], s)  # 修改索引
                    # out_path_prob = os.path.join(a_dir,
                    #                              'pred_' + test_name[k].replace(".png", "_s_" + str(s) + "_prob.png"))
                    # imageio.imsave(out_path_prob, prob_k)
                    # print('存储概率图：', 'pred_' + test_name[k].replace(".png", "_s_" + str(s) + "_prob.png"))

def save_model_distributions(model):
    dir_name = 'distributions'
    dir_path = os.path.join(globals.config['logging']['experiment_epoch_folder'], dir_name)
    os.makedirs(dir_path, exist_ok=True)
    method = globals.config['model']['method']
    if method == 'pionono':
        mu = model.z.posterior_mu.cpu().detach().numpy()
        covtril = model.z.posterior_covtril.cpu().detach().numpy()
        cov = np.zeros_like(covtril)
        for i in range(len(model.annotators)):
            cov[i] = np.matmul(covtril[i], covtril[i].transpose())
            np.savetxt(os.path.join(dir_path, "mu_" + str(i) + ".csv" ), np.round(mu[i], 4), delimiter=",", fmt="%.3f")
            np.savetxt(os.path.join(dir_path, "cov_" + str(i) + ".csv" ), np.round(cov[i], 4) , delimiter=",", fmt="%.3f")
        plot_and_save_distributions(mu, cov, dir_path)

def plot_and_save_distributions(mu_list, cov_list, dir_path):
    plt.figure()
    # plt.style.use('seaborn-dark')
    # plt.rcParams['figure.figsize'] = 14, 14
    no_annotators = mu_list.shape[0]

    twodim_mu_list = np.zeros(shape=[no_annotators, 2])
    twodim_cov_list = np.zeros(shape=[no_annotators, 2, 2])
    for i in range(no_annotators):
        twodim_mu_list[i] = mu_list[i][0:2]
        twodim_cov_list[i] = cov_list[i][0:2, 0:2]

    # Initializing the random seed
    random_seed = 0

    lim = np.max(np.abs(twodim_mu_list)) * 1.5 + np.max(np.abs(twodim_cov_list))
    x = np.linspace(- lim, lim, num=100)
    y = np.linspace(- lim, lim, num=100)
    X, Y = np.meshgrid(x, y)

    pdf_list = []

    for i in range(no_annotators):
        mean = twodim_mu_list[i]
        cov = twodim_cov_list[i]
        distr = multivariate_normal(cov=cov, mean=mean,
                                    seed=random_seed)
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
        pdf_list.append(pdf)

   # Plotting contour plots
    annotators = globals.config['data']['train']['masks']
    # colors = list(mcolors.TABLEAU_COLORS.keys())
    legend_list =[]
    for idx, val in enumerate(pdf_list):
        contourline = np.max(val) * (3/4)
        # cntr = plt.contour(X, Y, val, levels=[contourline], colors=colors[idx], alpha=0.7)
        cntr = plt.contour(X, Y, val, levels=[contourline], alpha=0.7)
        h, _ = cntr.legend_elements()
        legend_list.append(h[0])
    plt.legend(legend_list, annotators)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "dist_plot.jpg" ))
    plt.close()


def save_crowd_images(test_imgs:torch.Tensor, gt_pred: np.array, test_preds: np.array, test_labels: np.array, test_name: np.array, annotator, cm):
    visual_dir = 'qualitative_results/' + "train_crowd"
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    test_preds = np.asarray(test_preds, dtype=np.uint8)
    test_labels = np.asarray(test_labels, dtype=np.uint8)

    # print("test name ", test_name)
    out_path = os.path.join(dir, 'img_' + test_name)
    save_image(test_imgs, out_path)

    test_pred_rgb = convert_classes_to_rgb(test_preds)
    out_path = os.path.join(dir, annotator + '_pred_' + test_name)
    imageio.imsave(out_path, test_pred_rgb)

    gt_pred_rgb = convert_classes_to_rgb(gt_pred)
    out_path = os.path.join(dir, 'gt_pred_' + test_name)
    imageio.imsave(out_path, gt_pred_rgb)

    test_label_rgb = convert_classes_to_rgb(test_labels)
    out_path = os.path.join(dir, annotator + '_gt_' + test_name)
    imageio.imsave(out_path, test_label_rgb)

    cm = cm.detach().cpu().numpy()
    plt.matshow(cm)
    out_path = os.path.join(dir, annotator + '_matrix_' + test_name)
    plt.savefig(out_path)
    plt.close()


def save_image_color_legend():
    # visual_dir = 'qualitative_results/'
    dir = globals.config['logging']['experiment_folder']
    os.makedirs(dir, exist_ok=True)
    class_no = globals.config['data']['class_no']
    class_names = globals.config['data']['class_names']

    fig = plt.figure()

    size = 100

    for class_id in range(class_no):
        # out_img[size*class_id:size*(class_id+1),:,:] = convert_classes_to_rgb(np.ones(size,size,3)*class_id, size,size)
        out_img = convert_classes_to_rgb(np.ones(shape=[size,size])*class_id)
        ax = fig.add_subplot(1, class_no, class_id+1)
        ax.imshow(out_img)
        ax.set_title(class_names[class_id])
        ax.axis('off')
    plt.savefig(os.path.join(dir, 'legend.png'))
    plt.close()


def convert_classes_to_rgb(seg_classes):
    h = seg_classes.shape[0]
    w = seg_classes.shape[1]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    class_no = globals.config['data']['class_no']

    for class_id in range(class_no):
        # swap color channels because imageio saves images in RGB (not BGR)
        seg_rgb[:, :, 0][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][2]
        seg_rgb[:, :, 1][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][1]
        seg_rgb[:, :, 2][seg_classes == class_id] = CLASS_COLORS_BGR[class_id][0]

    return seg_rgb

def convert_std_to_rgb(std):
    h = std.shape[0]
    w = std.shape[1]
    var_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    values = np.clip(255 - (std * 2000), a_min=0, a_max=255).astype(int)
    var_rgb[:, :, 0]= values
    var_rgb[:, :, 1]= values
    var_rgb[:, :, 2]= values

    return var_rgb


def convert_probabilities_to_rgb(probabilities, type, a_dir, task_name, s):
    """"
    将概率转化为RGB
    """
    # 调试信息：打印输入的概率张量的形状和一些统计数据
    # print(f"Input probabilities shape: {probabilities.shape}")
    # print(f"Input probabilities min: {probabilities.min()}, max: {probabilities.max()}, mean: {probabilities.mean()}")

    h, w, class_no = probabilities.shape
    prob_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # CLASS_COLORS_BGR 转换为 RGB 格式
    CLASS_COLORS_RGB = [
        [CLASS_COLORS_BGR[i][2], CLASS_COLORS_BGR[i][1], CLASS_COLORS_BGR[i][0]]
        for i in range(len(CLASS_COLORS_BGR))
    ]

    # 确保 CLASS_COLORS_RGB 包含所有类别的颜色
    if class_no > len(CLASS_COLORS_RGB):
        raise ValueError(f"类别数量超过预定义颜色数量: {class_no} > {len(CLASS_COLORS_RGB)}")

    for class_id in range(class_no):
        color = np.array(CLASS_COLORS_RGB[class_id], dtype=np.uint8)
        class_prob = probabilities[:, :, class_id]

        # 调试信息：打印每个类别的概率统计数据
        # print(f"Class {class_id} probabilities min: {class_prob.min()}, max: {class_prob.max()}, mean: {class_prob.mean()}")

        # 将概率值归一化到 0-255
        prob_intensity = (class_prob * 255).astype(np.uint8)

        # 保存每个类别的概率图像
        if type == 'type1':
            class_prob_image_path = os.path.join(a_dir, 'pred_' + task_name.replace(".png", "_class" + str(class_id) + "_prob.png"))
        elif type == 'type2':
            class_prob_image_path = os.path.join(a_dir, 'pred_' + task_name.replace(".png", "_class" + str(class_id) + "_mean_prob.png"))
        else:
            class_prob_image_path = os.path.join(a_dir, 'pred_' + task_name.replace(".png", "_s_" + str(s) + "_class" + str(class_id) + "_prob.png"))

        imageio.imsave(class_prob_image_path, prob_intensity)
        # print(f"Saved class {class_id} probability image to {class_prob_image_path}")

        # 调试信息：打印归一化后的概率强度统计数据
        # print(f"Class {class_id} prob_intensity min: {prob_intensity.min()}, max: {prob_intensity.max()}, mean: {prob_intensity.mean()}")

        # 将颜色和强度应用到结果图像前打印调试信息
        # print(f"Applying color {color} to class probability map {class_id} with intensity stats: min {prob_intensity.min()}, max {prob_intensity.max()}, mean {prob_intensity.mean()}")

        # 将颜色和强度应用到结果图像
        for i in range(3):
            # 增加调试信息，检查每个通道的累加过程
            before_update = prob_rgb[:, :, i].copy()
            prob_rgb[:, :, i] = np.clip(prob_rgb[:, :, i] + (color[i] * class_prob).astype(np.uint8), 0, 255)
            # print(f"Channel {i}, Class {class_id}, min before: {before_update.min()}, max before: {before_update.max()}")
            # print(f"Channel {i}, Class {class_id}, min after: {prob_rgb[:, :, i].min()}, max after: {prob_rgb[:, :, i].max()}")

    # 调试信息：打印最终生成的RGB图像的统计数据
    # print(f"Output RGB image min: {prob_rgb.min()}, max: {prob_rgb.max()}, mean: {prob_rgb.mean()}")

    return prob_rgb


def save_results(results):
    results_dir = 'quantitative_results'
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], results_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'results.csv')
    with open(out_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])


def save_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    seg_model_ave_grads = []
    seg_model_max_grads = []
    seg_model_layers = []
    fcomb_ave_grads = []
    fcomb_max_grads = []
    fcomb_layers = []
    z_ave_grads = []
    z_max_grads = []
    z_layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            if 'seg_model' in n or 'unet' in n:
                seg_model_layers.append(n)
                seg_model_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                seg_model_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
            elif 'head' in n or 'fcomb' in n:
                fcomb_layers.append(n)
                fcomb_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                fcomb_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
            else:
                z_layers.append(n)
                z_ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                z_max_grads.append(p.grad.abs().max().cpu().detach().numpy())
        # print(n)
    plot_gradients(seg_model_ave_grads, seg_model_max_grads, seg_model_layers, name='gradients_seg_model.jpg')
    plot_gradients(fcomb_ave_grads, fcomb_max_grads, fcomb_layers, name='gradients_fcomb.jpg')
    plot_gradients(z_ave_grads, z_max_grads, z_layers, name='gradients_z.jpg')

def plot_gradients(ave_grads, max_grads, layers, name):
    foldername = 'gradients'
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=2, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=2, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=3, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    dir = os.path.join(globals.config['logging']['experiment_epoch_folder'], foldername)
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, name)
    plt.savefig(path)
    plt.close()
