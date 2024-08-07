import random
import numpy as np
import torch
from tqdm import tqdm
from utils.general_utils import heatmaps_to_coordinates
import wandb
from utils.testing import batch_epe_calculation
import torchmetrics as metrics
import torch.nn as nn
import torch
import os
import json
import cv2 as cv2
from utils.testing import (
    batch_epe_calculation,
    batch_auc_calculation,
    batch_pck_calculation,
)

from constants import action_to_verb
# FPHA_SCALE =  (1920, 1080)
FPHA_SCALE = (1280, 720)
FREIHAND = True
# DEBUG = True
DEBUG = False
DEBUG_SAMPLES = 10
softmax = nn.Softmax(dim=1)


class TrainerH2O:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, config: dict, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation

        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epe = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.epochs = config.max_epochs
        self.device = config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.wandb_logger = wandb_logger
        self.best_epe = 10000000

        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val EPE={}, Val Acc: {}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    np.round(self.epe["val"][-1], 10),
                    np.round(self.acc["val"][-1], 10)
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            save_best_model(model=self.model, run_name=self.run_name, new_value=np.round(
                self.epe["val"][-1], 10), best_value=self.best_epe, save_on_type="smaller_than")

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger:
                self.wandb_logger.log({"train_loss": np.round(
                    self.loss["train"][-1], 10), "val_loss": np.round(self.loss["val"][-1], 10), "val_epe": np.round(self.epe["val"][-1], 10),
                    "val_hand_acc": np.round(self.acc["val"][-1], 10)})

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'{self.run_name}_final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        activation = nn.Sigmoid()

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data["img"].to(self.device).type(torch.cuda.FloatTensor)

            labels_left = data['heatmap_left'].to(self.device)
            labels_right = data['heatmap_right'].to(self.device)

            labels_left_flag = data["left_hand_flag"].to(self.device)
            labels_right_flag = data["right_hand_flag"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(inputs)

            pred_left_flag = preds['left_handness']
            pred_right_flag = preds['right_handness']
            pred_left = preds['left_2D_pose']
            pred_right = preds['right_2D_pose']

            left_batch_mask = torch.argmax(
                activation(pred_left_flag), dim=1)
            right_batch_mask = torch.argmax(
                activation(pred_right_flag), dim=1)

            loss = self.criterion(pred_left_flag, pred_right_flag,
                                  labels_left_flag, labels_right_flag,
                                  pred_left, pred_right, labels_left, labels_right, batch_left=left_batch_mask, batch_right=right_batch_mask)

            loss.backward()

            self.optimizer.step()

            running_loss.append(loss.item())

            if DEBUG and i >= DEBUG_SAMPLES:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.eval()
        running_loss = []
        running_epe = []
        acc_lst = []

        Accuracy = metrics.Accuracy(
            task="multiclass", num_classes=2).to(self.device)

        activation = nn.Sigmoid()

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, 0)):

                inputs = data["img"].to(self.device).type(
                    torch.cuda.FloatTensor)

                labels_left = data['heatmap_left'].to(self.device)
                labels_right = data['heatmap_right'].to(self.device)
                labels_left_flag = data["left_hand_flag"].to(self.device)
                labels_right_flag = data["right_hand_flag"].to(self.device)

                preds = self.model(inputs)

                pred_left_flag = preds['left_handness']
                pred_right_flag = preds['right_handness']
                pred_left = preds['left_2D_pose']
                pred_right = preds['right_2D_pose']

                left_batch_mask = torch.argmax(
                    activation(pred_left_flag), dim=1)
                right_batch_mask = torch.argmax(
                    activation(pred_right_flag), dim=1)

                loss = self.criterion(pred_left_flag, pred_right_flag,
                                      labels_left_flag, labels_right_flag,
                                      pred_left, pred_right, labels_left, labels_right, batch_left=left_batch_mask, batch_right=right_batch_mask)

                acc_left = Accuracy(pred_left_flag, labels_left_flag)
                acc_right = Accuracy(pred_right_flag, labels_right_flag)

                acc_left = acc_left*pred_left_flag.shape[0]
                acc_right = acc_right*pred_right_flag.shape[0]
                acc_lst.append(0.5*(acc_left+acc_right))

                pred_left_np = heatmaps_to_coordinates(
                    pred_left.cpu().detach().numpy(), img_size=pred_left.shape[3])
                pred_right_np = heatmaps_to_coordinates(
                    pred_right.cpu().detach().numpy(), img_size=pred_right.shape[3])

                labels_left_np = data['keypoints_left'].cpu().detach().numpy()
                labels_right_np = data['keypoints_right'].cpu(
                ).detach().numpy()

                epe_left = self.__get_epe(
                    pred_left_np, labels_left_np, left_batch_mask.cpu().detach().numpy())
                epe_right = self.__get_epe(
                    pred_right_np, labels_right_np, right_batch_mask.cpu().detach().numpy())

                temp_epe = epe_left + epe_right

                running_epe.append(temp_epe)
                running_loss.append(loss.item())

                if DEBUG and i >= DEBUG_SAMPLES:
                    break

            epoch_loss = np.mean(running_loss)
            epe_loss = np.mean(running_epe)

            accuracy = sum(acc_lst)/len(dataloader.dataset)

            self.loss["val"].append(epoch_loss)
            self.epe["val"].append(epe_loss)
            self.acc["val"].append(accuracy.cpu().numpy())

    def __get_epe(self, preds, labels, batch_mask):

        preds = preds * (1280, 720)
        labels = labels * (1280, 720)
        return batch_epe_calculation(preds, labels, batch_mask)


class TrainerFreiHAND:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, config: dict, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation

        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epe = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.epochs = config.max_epochs
        self.device = config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.wandb_logger = wandb_logger
        self.best_epe = 10000000
        self.one_batch_flag = config.one_batch_flag

        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val EPE={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    np.round(self.epe["val"][-1], 10),
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            # self.save_best_model(val = np.round(self.epe["val"][-1], 10))
            save_best_model(model=self.model, run_name=self.run_name, new_value=np.round(
                self.epe["val"][-1], 10), best_value=self.best_epe, save_on_type="smaller_than")

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger:
                self.wandb_logger.log({"train_loss": np.round(
                    self.loss["train"][-1], 10), "val_loss": np.round(self.loss["val"][-1], 10), "val_epe": np.round(self.epe["val"][-1], 10)
                })

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'{self.run_name}_final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data["img"].to(self.device)
            labels = data["heatmaps"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss.append(loss.item())

            if DEBUG and i >= DEBUG_SAMPLES:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.eval()
        running_loss = []
        running_epe = []

        with torch.no_grad():

            for i, data in enumerate(tqdm(dataloader, 0)):

                inputs = data["img"].to(self.device)
                labels = data["heatmaps"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

                pred_keypoints_np = heatmaps_to_coordinates(
                    outputs.cpu().detach().numpy(), img_size=outputs.shape[3])

                labels_keypoints_np = data['keypoints'].cpu().detach().numpy()

                epe = self.__get_epe(
                    pred_keypoints_np, labels_keypoints_np)

                running_epe.append(epe)
                running_loss.append(loss.item())

                if DEBUG and i >= DEBUG_SAMPLES:
                    break

            epoch_loss = np.mean(running_loss)
            epe_loss = np.mean(running_epe)

            self.loss["val"].append(epoch_loss)
            self.epe["val"].append(epe_loss)

    def __get_epe(self, preds, labels, batch_mask=None):

        preds = preds * (128, 128)
        labels = labels * (128, 128)
        return batch_epe_calculation(preds, labels, batch_mask)


class Trainer_FPHA:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, config: dict, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation

        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epe = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.epochs = config.max_epochs
        self.device = config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.wandb_logger = wandb_logger
        self.best_epe = 10000000
        self.one_batch_flag = config.one_batch_flag
        self.obj_accuracy = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val EPE={}, Obj Acc={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    np.round(self.epe["val"][-1], 10),
                    self.obj_accuracy[-1],
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step()

            save_best_model(model=self.model, run_name=self.run_name, new_value=np.round(
                self.epe["val"][-1], 10), best_value=self.best_epe, save_on_type="smaller_than")

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger:
                self.wandb_logger.log({"train_loss": np.round(
                    self.loss["train"][-1], 10), "val_loss": np.round(self.loss["val"][-1], 10), "val_epe": np.round(self.epe["val"][-1], 10),
                    "obj_acc": self.obj_accuracy[-1]
                })

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'{self.run_name}_final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data["img"].to(self.device).type(
                torch.cuda.FloatTensor)
            labels = data["heatmaps"].to(self.device).type(
                torch.cuda.FloatTensor)
            obj_label = data['obj'].to(self.device).type(
                torch.cuda.LongTensor)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):

                pred_obj, pred_pose = self.model(inputs)
                loss = self.criterion(pred_obj, pred_pose, obj_label, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            running_loss.append(loss.item())

            if DEBUG and i >= DEBUG_SAMPLES:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.eval()
        running_loss = []
        running_epe = []
        acc_lst = []

        accuracy = metrics.Accuracy(
            task="multiclass", num_classes=27).to(self.device)

        acc_lst = []

        with torch.no_grad():

            for i, data in enumerate(tqdm(dataloader, 0)):

                inputs = data['img'].to(self.device)
                labels = data["heatmaps"].to(self.device)
                obj_label = data['obj'].to(self.device)
                pred_obj, pred_pose = self.model(inputs)
                kpts = data['keypoints']

                loss = self.criterion(pred_obj, pred_pose, obj_label, labels)

                running_loss.append(loss.item())

                pred_keypoints_np = heatmaps_to_coordinates(
                    pred_pose.cpu().detach().numpy(), img_size=pred_pose.shape[3]) * FPHA_SCALE

                labels_keypoints_np = kpts.cpu(
                ).detach().numpy() * FPHA_SCALE

                # Calculate euclidian distance between pred_keypoints_np of shape (64,21,2) and labels_keypoints_np of shape (64,21,2)
                epe = self.__get_epe(
                    pred_keypoints_np, labels_keypoints_np)

                running_epe.append(epe)
                running_loss.append(loss.item())

                acc_obj = accuracy(pred_obj, obj_label).cpu().detach().numpy()
                acc_lst.append(acc_obj)

                if DEBUG and i >= DEBUG_SAMPLES:
                    break

            epoch_loss = np.mean(running_loss)
            epe_loss = np.mean(running_epe)

            self.loss["val"].append(epoch_loss)
            self.epe["val"].append(epe_loss)
            accuracy = np.mean(acc_lst)
            self.obj_accuracy.append(accuracy)

    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.eval()
        running_loss = []
        running_epe = []
        acc_lst = []

        accuracy = metrics.Accuracy(
            task="multiclass", num_classes=27).to(self.device)

        acc_lst = []

        pths_list = []
        pose_list = []
        obj_list = []
        confidence_list = []
        pck_2d = []
        auc_lst = []
        correct = 0
        with torch.no_grad():

            for i, data in enumerate(tqdm(dataloader, 0)):

                pths = data['img_pth']

                pths_list += pths

                inputs = data["img"].to(self.device)
                labels = data["heatmaps"].to(self.device)
                obj_label = data['obj'].to(self.device)

                pred_obj, pred_pose = self.model(inputs)

                loss = self.criterion(pred_obj, pred_pose, obj_label, labels)
                running_loss.append(loss.item())

                pred_keypoints_np = heatmaps_to_coordinates(
                    pred_pose.cpu().detach().numpy(), img_size=pred_pose.shape[3])

                labels_keypoints_np = data['keypoints'].cpu(
                ).detach().numpy()

                # Check if all values in numpy array labels_keypoints_np is in range between 0 and 1
                if np.all((0 <= labels_keypoints_np) & (labels_keypoints_np <= 1)):

                    epe = self.__get_epe(
                        pred_keypoints_np * FPHA_SCALE, labels_keypoints_np * FPHA_SCALE)

                    pck_temp = batch_pck_calculation(
                        pred_keypoints_np * FPHA_SCALE, labels_keypoints_np * FPHA_SCALE, treshold=0.2, mask=None, normalize=None)
                    pck_2d.append(pck_temp)
                    correct += 1

                    auc = batch_auc_calculation(
                        pred_keypoints_np * FPHA_SCALE, labels_keypoints_np * FPHA_SCALE, num_step=20, mask=None)
                    auc_lst.append(auc)

                    running_epe.append(epe)
                    running_loss.append(loss.item())

                pose_list += pred_keypoints_np.tolist()
                # get confidence of predictions

                pred_obj = softmax(pred_obj)
                # pred_obj = torch.argmax(pred_obj, dim=1)
                obj_confidence, pred_obj = torch.max(pred_obj, dim=1)
                obj_list += pred_obj.cpu().detach().tolist()
                confidence_list += obj_confidence.cpu().detach().tolist()
                acc_obj = accuracy(pred_obj, obj_label).cpu().detach().numpy()
                acc_lst.append(acc_obj)

            epoch_loss = np.mean(running_loss)
            epe_loss = np.mean(running_epe)

            self.loss["val"].append(epoch_loss)
            self.epe["val"].append(epe_loss)
            accuracy = np.mean(acc_lst)
            self.obj_accuracy.append(accuracy)

            output_dict_pose = {pths_list[i]: pose_list[i]
                                for i in range(len(pose_list))}

            output_dict_obj = {pths_list[i]: obj_list[i]
                               for i in range(len(pose_list))}

            output_dict_confidence = {pths_list[i]: confidence_list[i]
                                      for i in range(len(pose_list))}

            with open("pred_poses_asdf2_train.txt", "w") as fp:
                json.dump(output_dict_pose, fp)  # encode dict into JSON

            with open("pred_obj.txt", "w") as fp:
                json.dump(output_dict_obj, fp)  # encode dict into JSON

            with open("pred_confidence_tuple.txt", "w") as fp:
                json.dump(output_dict_confidence, fp)

            print('Loss: ', epoch_loss)
            print('EPE: ', epe_loss)
            print('PCK: ', np.mean(np.array(pck_2d)))
            print('AUC: ', np.mean(np.array(auc)))
            print('Object Acc: ', accuracy)
            print(
                f'Correct: {correct} out of {len(dataloader.dataset)}, {correct/len(dataloader.dataset)}%')

    def __get_epe(self, preds, labels, batch_mask=None):

        return batch_epe_calculation(preds, labels, batch_mask)


class TrainerFPHAAction:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, training_config, model_config, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation
        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": [], "test": []}
        self.epochs = training_config.max_epochs
        self.device = training_config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = training_config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.best_acc_val = 0
        self.wandb_logger = wandb_logger
        self.acc_val = []
        self.acc_test = []

        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)

            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val Acc={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    self.acc_val[-1],
                )
            )
            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step()

            self.best_acc_val = save_best_model(self.model, self.run_name,
                                                self.acc_val[-1], self.best_acc_val, 'greater_than')

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            self.wandb_logger.log({"acc": self.acc_val[-1], "train_loss": np.round(
                self.loss["train"][-1], 10), "val_loss": np.round(self.loss["val"][-1], 10), "best_acc": self.best_acc_val})

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data['keypoints'].to(self.device)
            labels = data['action_label'].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss.append(loss.item()*labels.shape[0])

        epoch_loss = sum(running_loss)/len(dataloader.dataset)
        self.loss["train"].append(epoch_loss)

    def test_model(self, test_dataloader: torch.utils.data.DataLoader):
        acc = []
        self.__evaluation(dataloader=test_dataloader,
                          loss_list=self.loss["test"], acc_list=acc)
        print('Acc: ', np.mean(acc), ' Loss: ', self.loss["test"][-1])

    def __evaluation(self, dataloader: torch.utils.data.DataLoader, loss_list, acc_list):

        self.model.eval()
        running_loss = []
        Accuracy = metrics.Accuracy(
            task="multiclass", num_classes=45).to(self.device)
        acc_lst = []

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                inputs = data['keypoints'].to(self.device)
                labels = data['action_label'].to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item()*labels.shape[0])
                prob = outputs.softmax(dim=1)
                acc = Accuracy(prob, labels)
                acc_lst.append(acc*labels.shape[0])

            epoch_loss = sum(running_loss)/len(dataloader.dataset)
            loss_list.append(epoch_loss)
            accuracy = sum(acc_lst)/len(dataloader.dataset)
            acc_list.append(accuracy.cpu().numpy())

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.__evaluation(dataloader=dataloader,
                          loss_list=self.loss["val"], acc_list=self.acc_val)


class TrainerH2OAction:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, training_config, model_config, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation
        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": [], "test": []}
        self.epochs = training_config.max_epochs
        self.device = training_config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = training_config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.best_acc_val = 0
        self.wandb_logger = wandb_logger
        self.acc_val = []
        self.acc_test = []

        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val Acc={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    self.acc_val[-1],
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step()

            self.best_acc_val = save_best_model(model=self.model, run_name=self.run_name, new_value=np.round(
                self.acc_val[-1], 10), best_value=self.best_acc_val, save_on_type="greater_than")

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger:
                self.wandb_logger.log(
                    {"acc": self.acc_val[-1],
                     "train_loss": np.round(
                        self.loss["train"][-1], 10),
                        "val_loss": np.round(self.loss["val"][-1], 10),
                        "best_acc": self.best_acc_val,
                     })

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data['positions'].to(
                self.device).type(torch.cuda.FloatTensor)

            labels = data['action_label'].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optimizer.step()
            running_loss.append(loss.item()*labels.shape[0])

        epoch_loss = sum(running_loss)/len(dataloader.dataset)
        self.loss["train"].append(epoch_loss)

    def test_model(self, test_dataloader: torch.utils.data.DataLoader):
        self.__evaluation(dataloader=test_dataloader,
                          loss_list=self.loss["test"], acc_list=self.acc_test)
        print('Acc: ', self.acc_test[-1], ' Loss: ', self.loss["test"][-1])

    def __evaluation(self, dataloader: torch.utils.data.DataLoader, loss_list, acc_list):

        self.model.eval()
        running_loss = []
        Accuracy = metrics.Accuracy(
            task="multiclass", num_classes=37).to(self.device)
        Accuracy2 = metrics.Accuracy(
            task="multiclass", num_classes=37).to(self.device)
        acc_lst = []
        acc_verbs = []
        y_true_all = np.array([])
        y_pred_all = np.array([])

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                inputs = data['positions'].to(
                    self.device).type(torch.cuda.FloatTensor)
                labels = data['action_label'].to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item()*labels.shape[0])
                prob = softmax(outputs)

                indxs = torch.argmax(prob, dim=1)
                acc = Accuracy(prob, labels)

                verb_preds = torch.tensor(
                    [action_to_verb[int(x)] for x in indxs])
                verb_labels = torch.tensor(
                    [action_to_verb[int(x)] for x in labels])

                y_true_all = np.append(y_true_all, verb_labels)
                y_pred_all = np.append(y_pred_all, verb_preds)

                acc_verb = Accuracy2(verb_preds, verb_labels)
                acc_verbs.append(acc_verb*labels.shape[0])

                acc_lst.append(acc*labels.shape[0])
            epoch_loss = sum(running_loss)/len(dataloader.dataset)
            loss_list.append(epoch_loss)
            accuracy = sum(acc_lst)/len(dataloader.dataset)
            accuracy_verb = sum(acc_verbs)/len(dataloader.dataset)
            print('Acc verb: ', accuracy_verb.item())
            acc_list.append(accuracy.cpu().numpy())

    def test_h2o(self, dataloader: torch.utils.data.DataLoader):

        self.model.eval()

        h2o_test_results = {
            "modality": "hand+obj",
        }
        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                inputs = data['positions'].to(
                    self.device).type(torch.cuda.FloatTensor)

                outputs = self.model(inputs)
                prob = softmax(outputs)
                # In prob of shape 64, 36 pick the highest value in each row

                indxs = torch.argmax(prob, dim=1)

                for key, value in zip(data['action_id'], indxs):
                    h2o_test_results[int(key)] = int(value)

            print(f'Len of dict: {len(h2o_test_results)}')
            with open("action_labels.json", "w") as fp:
                json.dump(h2o_test_results, fp)
            os.system('rm answear.zip')
            os.system('zip answear.zip action_labels.json')
            return h2o_test_results

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.__evaluation(dataloader=dataloader,
                          loss_list=self.loss["val"], acc_list=self.acc_val)


def save_best_model(model, run_name, new_value: float, best_value, save_on_type: str):
    """
    Saves best model
    Args:
        val_loss (float): Current validation loss
        epoch (int): Current epoch
    """

    # Check if folder named wandb.run.name exists, if not, create the folder
    if not os.path.exists(f'checkpoints/{run_name}'):
        os.makedirs(f'checkpoints/{run_name}')

    best_value_ret = best_value
    # When higher value:
    if save_on_type == 'greater_than':

        if new_value >= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')

    else:
        if new_value <= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')

    return best_value_ret
