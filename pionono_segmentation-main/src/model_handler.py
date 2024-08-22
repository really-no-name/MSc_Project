import os
import torch
import mlflow
import warnings
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

import utils.globals as globals
from timeit import default_timer as timer

from utils.saving import save_model, save_results, save_test_images, save_image_color_legend, save_crowd_images, \
    save_grad_flow, save_test_image_variability, save_model_distributions
from utils.test_helpers import segmentation_scores
from utils.mlflow_logger import log_results, log_results_list, probabilistic_model_logging, set_epoch_output_dir,\
    set_test_output_dir, log_artifact_folder
from utils.initialize_optimization import init_optimization
from utils.initialize_model import init_model
import matplotlib.pyplot as plt

eps=1e-7


# class ModelHandler:
#     """
#     The model handler initializes, trains, and tests the different models.
#     """
#     def __init__(self, annotators, predict_mode=False):
#         self.train_img_vis = []
#         self.train_label_vis = []
#         self.train_pred_vis = []
#         self.train_img_name = []
#         self.epoch = 0
#         self.model = init_model(annotators)
#         self.model.cuda()
#         # -----------------XU
#         self.predict_mode = predict_mode
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.annotators = annotators
#         if not self.predict_mode:
#             self.model = init_model(annotators).to(self.device)
#         else:
#             self.model = None
#         # -----------------XU
#         if torch.cuda.is_available():
#             print('Running on GPU')
#             self.device = torch.device('cuda')
#         else:
#             warnings.warn("Running on CPU because no GPU was found!")
#             self.device = torch.device('cpu')
#
#     def train(self, trainloader, validate_data):
#         """
#         Training of one model with the given configuration.
#         """
#         config = globals.config
#         model = self.model
#         epochs = config['model']['epochs']
#         batch_s = config['model']['batch_size']
#         print('model epochs:', epochs)
#         print('model batch_size:', batch_s)
#
#         save_image_color_legend()
#
#         optimizer, loss_fct = init_optimization(model)
#
#         # Training loop
#         for i in range(0, epochs):
#             print('\nEpoch: {}'.format(i))
#             model.train()
#             set_epoch_output_dir(i)
#             self.epoch = i
#
#             # Training in batches
#             for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
#                 # Loading data to GPU
#                 images = images.cuda().float()
#                 labels = labels.cuda().long()
#                 ann_ids = ann_ids.cuda().float()
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 _, labels = torch.max(labels, dim=1)
#                 loss, y_pred = model.train_step(images, labels, loss_fct, ann_ids)
#
#                 if j % int(config['logging']['interval']) == 0:
#                     print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
#                     self.log_training_metrics(y_pred, labels, loss, model, i * len(trainloader) * batch_s + j)
#                 # self.store_train_imgs(imagename, images, labels, y_pred)
#
#                 if config['model']['method'] == 'conf_matrix' and i == config['model']['conf_matrix_config']['activate_min_trace_epoch']:  # 10 for cr_image_dice // 5 rest of the methods
#                     optimizer = model.activate_min_trace()
#
#                 # Backprop
#                 if not torch.isnan(loss):
#                     loss.backward()
#                     optimizer.step()
#
#             mlflow.log_metric('finished_epochs', self.epoch + 1, int((i + 1) * len(trainloader) * batch_s))
#
#             if (i + 1) % int(config['logging']['artifact_interval']) == 0:
#                 val_results = self.evaluate(validate_data, mode='val')
#                 log_results_list(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
#                 save_grad_flow(model.named_parameters())
#                 self.save_train_imgs()
#
#             # LR decay
#             if i > config['model']['lr_decay_after_epoch']:
#                 for g in optimizer.param_groups:
#                     g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])
#         save_model(model)
#
#     def test(self, test_data):
#         """
#         Test a model with the given configuration.
#         """
#         set_test_output_dir()
#         save_image_color_legend()
#         results = self.evaluate(test_data)
#         log_results_list(results, mode='test', step=None)
#         log_artifact_folder()
#
#     def evaluate(self, data, mode='test'):
#         """
#         The evaluation function to obtain the model metric over a defined set of images with labels (data).
#         """
#         config = globals.config
#         class_no = config['data']['class_no']
#         vis_images = config['data']['visualize_images'][mode]
#
#         model = self.model
#
#         device = self.device
#         model.eval()
#         annotator_list = data[0]
#         loader_list = data[1]
#         save_model_distributions(model)
#
#         with torch.no_grad():
#             results_list = []
#
#             for e in range(len(loader_list)):
#                 labels = []
#                 preds = []
#                 start_time = timer()
#                 for j, (test_img, test_label, test_name, ann_id) in enumerate(loader_list[e]):
#                     test_img = test_img.to(device=device, dtype=torch.float32)
#                     if config['model']['method'] == 'pionono':
#                         model.forward(test_img)
#                         if 'STAPLE' in annotator_list[e] or 'MV' in annotator_list[e] or 'expert' in annotator_list[e] or config['model']['pionono_config']['always_goldpred']:
#                             test_pred, _ = model.get_gold_predictions()
#                         else:
#                             test_pred = model.sample(use_z_mean=True, annotator_ids=ann_id, annotator_list=annotator_list)
#                     elif config['model']['method'] == 'prob_unet':
#                         model.forward(test_img, None, training=False)
#                         test_pred = model.get_gold_predictions()
#                     else:
#                         test_pred = model(test_img)
#                     _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)
#                     test_pred_np = test_pred.cpu().detach().numpy()
#                     test_label = test_label.cpu().detach().numpy()
#                     test_label = np.argmax(test_label, axis=1)
#
#                     preds.append(test_pred_np.astype(np.int8).copy().flatten())
#                     labels.append(test_label.astype(np.int8).copy().flatten())
#
#                     # We only want to execute the code below once for each image.
#                     if e == 0:
#                         if self.epoch % int(config['logging']['artifact_interval']) == 0 or mode == 'test':
#                             for k in range(len(test_name)):
#                                 if test_name[k] in vis_images or vis_images == 'all':
#                                     img = test_img[k]
#                                     save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)
#                                     save_test_image_variability(model, test_name, k, mode)
#                 end_time = timer()
#                 print('Average inference time: ' + str((end_time - start_time)/len(loader_list[e])))
#                 preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
#                 labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()
#                 if e == 0:
#                     shortened = False
#                 else:
#                     shortened = True
#                 results = self.get_results(preds, labels, shortened)
#
#                 print('RESULTS for ' + mode + ' Annotator: ' + str(annotator_list[e]))
#                 print(results)
#                 results_list.append(results)
#
#         log_artifact_folder()
#         return results_list
#
#     def get_results(self, pred, label, shortened=False):
#         if torch.is_tensor(pred):
#             pred = pred.cpu().detach().numpy().copy().flatten()
#         if torch.is_tensor(label):
#             label = label.cpu().detach().numpy().copy().flatten()
#
#         results = segmentation_scores(label, pred, shortened)
#
#         return results
#
#     def log_training_metrics(self, y_pred, labels, loss, model, step):
#         config = globals.config
#         _, y_pred = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
#         mlflow.log_metric('loss',float(loss.cpu().detach().numpy()), step=step)
#         train_results = self.get_results(y_pred, labels)
#         log_results(train_results, mode='train', step=step)
#         probabilistic_model_logging(model, step)
#
#     def store_train_imgs(self, imagenames, images, labels, y_pred):
#         """
#         For debugging, images can be saved.
#         """
#         config = globals.config
#         vis_train_images = config['data']['visualize_images']['train']
#
#         for k in range(len(imagenames)):
#             if imagenames[k] in vis_train_images:
#                 _, y_pred_argmax = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
#                 self.train_img_vis.append(images[k])
#                 self.train_label_vis.append(labels[k].cpu().detach().numpy())
#                 self.train_pred_vis.append(y_pred_argmax[k].cpu().detach().numpy())
#                 self.train_img_name.append(imagenames[k])
#
#     def save_train_imgs(self):
#         for i in range(len(self.train_img_vis)):
#             save_test_images(self.train_img_vis[i], self.train_pred_vis[i],
#                              self.train_label_vis[i], self.train_img_name[i], 'train')
#         self.train_img_vis = []
#         self.train_label_vis = []
#         self.train_pred_vis = []
#         self.train_img_name = []
#
#     # ------------------XU-----------------------
#     def load_model(self, model_path):
#         # Add logic to load the model from the specified path
#         if torch.cuda.is_available():
#             self.model = torch.load(model_path)
#         else:
#             self.model = torch.load(model_path, map_location=torch.device('cpu'))
#
#         self.model.to(self.device)
#         self.model.eval()
#         print(f"Model loaded: {self.model}")
#
#     def predict(self, image_path):
#         # Add logic to predict the segmentation for the given image
#         image = Image.open(image_path)
#         # Preprocess image
#         input_tensor = self.preprocess_image(image)
#         input_tensor = input_tensor.unsqueeze(0).to(self.device)  # create a mini-batch as expected by the model
#         print(f"Input tensor: {input_tensor.shape}")
#
#         annotator_ids = torch.zeros(input_tensor.shape[0], dtype=torch.long).to(
#             self.device)  # Assuming single annotator ID for simplicity
#
#         with torch.no_grad():
#             output = self.model(input_tensor, annotator_ids=annotator_ids)
#             print(f"Model output: {output}")
#
#         # Post-process the output to get the segmentation map
#         result = self.postprocess_output(output)
#
#         # Save and visualize the result
#         self.save_result(result, image_path)
#         self.visualize_result(result)
#
#         return result
#
#     def preprocess_image(self, image):
#         # Add image preprocessing steps
#         preprocess = transforms.Compose([
#             transforms.Resize((256, 256)),  # Resize to the expected input size of the model
#             transforms.ToTensor(),  # Convert the image to a tensor
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
#         ])
#         return preprocess(image)
#
#     def postprocess_output(self, output):
#         # Add post-processing steps
#         if output is None:
#             raise ValueError("Model output is None")
#         return output.argmax(dim=1).squeeze().cpu().numpy()
#
#     def save_result(self, result, image_path):
#         # Save the result as an image file
#         result_image = Image.fromarray((result * 255).astype(np.uint8))
#         result_path = image_path.replace(".png", "_segmentation.png")
#         result_image.save(result_path)
#         print(f"Segmentation result saved to {result_path}")
#
#     def visualize_result(self, result):
#         # Visualize the result using matplotlib
#         plt.imshow(result, cmap='gray')
#         plt.title("Segmentation Result")
#         plt.axis("off")
#         plt.show()


class ModelHandler:
    """
    The model handler initializes, trains, and tests the different models.
    """

    def __init__(self, annotators, predict_mode=False):
        self.train_img_vis = []
        self.train_label_vis = []
        self.train_pred_vis = []
        self.train_img_name = []
        self.epoch = 0
        self.model = init_model(annotators)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, trainloader, validate_data):
        """
        Training of one model with the given configuration.
        """
        config = globals.config
        model = self.model
        epochs = config['model']['epochs']
        batch_s = config['model']['batch_size']

        save_image_color_legend()

        optimizer, loss_fct = init_optimization(model)

        # Training loop
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i))
            model.train()
            set_epoch_output_dir(i)
            self.epoch = i

            # Training in batches
            for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
                # Loading data to GPU
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                ann_ids = ann_ids.to(self.device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                _, labels = torch.max(labels, dim=1)
                loss, y_pred = model.train_step(images, labels, loss_fct, ann_ids)

                if j % int(config['logging']['interval']) == 0:
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
                    self.log_training_metrics(y_pred, labels, loss, model, i * len(trainloader) * batch_s + j)
                # self.store_train_imgs(imagename, images, labels, y_pred)

                if config['model']['method'] == 'conf_matrix' and i == config['model']['conf_matrix_config'][
                    'activate_min_trace_epoch']:  # 10 for cr_image_dice // 5 rest of the methods
                    optimizer = model.activate_min_trace()

                # Backprop
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            mlflow.log_metric('finished_epochs', self.epoch + 1, int((i + 1) * len(trainloader) * batch_s))

            if (i + 1) % int(config['logging']['artifact_interval']) == 0:
                val_results = self.evaluate(validate_data, mode='val')
                log_results_list(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
                save_grad_flow(model.named_parameters())
                self.save_train_imgs()

            # LR decay
            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])
        save_model(model)

    def test(self, test_data):
        """
        Test a model with the given configuration.
        """
        set_test_output_dir()
        save_image_color_legend()
        results = self.evaluate(test_data)
        log_results_list(results, mode='test', step=None)
        log_artifact_folder()

    def evaluate(self, data, mode='test'):
        """
        The evaluation function to obtain the model metric over a defined set of images with labels (data).
        """
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]

        model = self.model

        device = self.device
        model.eval()
        annotator_list = data[0]
        loader_list = data[1]
        save_model_distributions(model)

        with torch.no_grad():
            results_list = []

            for e in range(len(loader_list)):
                labels = []
                preds = []
                start_time = timer()
                for j, (test_img, test_label, test_name, ann_id) in enumerate(loader_list[e]):
                    test_img = test_img.to(device=device, dtype=torch.float32)
                    if config['model']['method'] == 'pionono':
                        model.forward(test_img)
                        if 'STAPLE' in annotator_list[e] or 'MV' in annotator_list[e] or 'expert' in annotator_list[
                            e] or config['model']['pionono_config']['always_goldpred']:
                            test_pred, _ = model.get_gold_predictions()
                        else:
                            test_pred = model.sample(use_z_mean=True, annotator_ids=ann_id,
                                                     annotator_list=annotator_list)
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

                    # We only want to execute the code below once for each image.
                    if e == 0:
                        if self.epoch % int(config['logging']['artifact_interval']) == 0 or mode == 'test':
                            for k in range(len(test_name)):
                                if test_name[k] in vis_images or vis_images == 'all':
                                    img = test_img[k]
                                    save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)
                                    save_test_image_variability(model, test_name, k, mode)
                end_time = timer()
                print('Average inference time: ' + str((end_time - start_time) / len(loader_list[e])))
                preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
                labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()
                if e == 0:
                    shortened = False
                else:
                    shortened = True
                results = self.get_results(preds, labels, shortened)

                print('RESULTS for ' + mode + ' Annotator: ' + str(annotator_list[e]))
                print(results)
                results_list.append(results)

        log_artifact_folder()
        return results_list

    def get_results(self, pred, label, shortened=False):
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy().copy().flatten()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy().copy().flatten()

        results = segmentation_scores(label, pred, shortened)

        return results

    def log_training_metrics(self, y_pred, labels, loss, model, step):
        config = globals.config
        _, y_pred = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
        mlflow.log_metric('loss', float(loss.cpu().detach().numpy()), step=step)
        train_results = self.get_results(y_pred, labels)
        log_results(train_results, mode='train', step=step)
        probabilistic_model_logging(model, step)

    def store_train_imgs(self, imagenames, images, labels, y_pred):
        """
        For debugging, images can be saved.
        """
        config = globals.config
        vis_train_images = config['data']['visualize_images']['train']

        for k in range(len(imagenames)):
            if imagenames[k] in vis_train_images:
                _, y_pred_argmax = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
                self.train_img_vis.append(images[k])
                self.train_label_vis.append(labels[k].cpu().detach().numpy())
                self.train_pred_vis.append(y_pred_argmax[k].cpu().detach().numpy())
                self.train_img_name.append(imagenames[k])

    def save_train_imgs(self):
        for i in range(len(self.train_img_vis)):
            save_test_images(self.train_img_vis[i], self.train_pred_vis[i],
                             self.train_label_vis[i], self.train_img_name[i], 'train')
        self.train_img_vis = []
        self.train_label_vis = []
        self.train_pred_vis = []
        self.train_img_name = []

    def predict(self, image_path):
        image = Image.open(image_path)
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        print(f"Input tensor: {input_tensor.shape}")

        annotator_ids = torch.zeros(input_tensor.shape[0], dtype=torch.long).to(self.device)

        with torch.no_grad():
            self.model.forward(input_tensor)
            output, uncertainty = self.model.get_gold_predictions()
            print(f"Model output: {output}")
            print(f"Uncertainty output: {uncertainty}")

        result = self.postprocess_output(output)
        uncertainty = self.postprocess_output(uncertainty)

        self.save_result(result, uncertainty, image_path)
        self.visualize_result(result, uncertainty)

        return result, uncertainty

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image)


    def postprocess_output(self, output):
        if output is None:
            raise ValueError("Model output is None")
        return output.argmax(dim=1).squeeze().cpu().numpy()

    def save_result(self, result, uncertainty, image_path):
        result_image = Image.fromarray((result * 255).astype(np.uint8))
        uncertainty_image = Image.fromarray((uncertainty * 255).astype(np.uint8))

        result_path = image_path.replace(".png", "_segmentation.png")
        uncertainty_path = image_path.replace(".png", "_uncertainty.png")

        result_image.save(result_path)
        uncertainty_image.save(uncertainty_path)

        print(f"Segmentation result saved to {result_path}")
        print(f"Uncertainty result saved to {uncertainty_path}")

    def visualize_result(self, result, uncertainty):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(result, cmap='gray')
        axs[0].set_title("Segmentation Result")
        axs[0].axis("off")

        axs[1].imshow(uncertainty, cmap='hot')
        axs[1].set_title("Uncertainty Result")
        axs[1].axis("off")

        plt.show()