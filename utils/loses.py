import torch.nn as nn
import numpy as np
import torch


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) loss function for semantic segmentation.

    Args:
        epsilon (float): A small value to avoid division by zero.

    Attributes:
        epsilon (float): A small value to avoid division by zero.

    Methods:
        _op_sum(x): Computes the sum of elements in the input tensor along the last two dimensions.
        forward(y_pred, y_true): Computes the IoU loss between the predicted and ground truth heatmaps.

    """

    def __init__(self, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def _op_sum(self, x):
        """
        Computes the sum of elements in the input tensor along the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sum of elements along the last two dimensions.

        """
        return x.sum(-1).sum(-1)

    def forward(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Computes the IoU loss between the predicted and ground truth heatmaps.

        Args:
            y_pred (np.array): Predicted heatmap.
            y_true (np.array): Ground truth heatmap.

        Returns:
            float: IoU loss.

        """
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.epsilon) / (union + self.epsilon)
        iou = torch.mean(iou)

        return 1 - iou

class EffHandEgoNetLoss(nn.Module):
    """
    This class defines the loss function for the EffHandEgoNet model.

    Args:
    - None

    Returns:
    - None
    """

    def __init__(self):
        """
        Initializes the loss function.

        Args:
        - None

        Returns:
        - None
        """
        super(EffHandEgoNetLoss, self).__init__()

        self.criterion_left_flag = nn.CrossEntropyLoss()
        self.criterion_right_flag = nn.CrossEntropyLoss()
        self.criterion_left = IoULoss()
        self.criterion_right = IoULoss()
        self.softmax = nn.Softmax()

    def forward(self, pred_left_flag, pred_right_flag,
                labels_left_flag, labels_right_flag, left_pred, right_pred, left_labels, right_labels, batch_left, batch_right):
        """
        Computes the loss for the given inputs.

        Args:
        - pred_left_flag (torch.Tensor): predicted left flag tensor
        - pred_right_flag (torch.Tensor): predicted right flag tensor
        - labels_left_flag (torch.Tensor): ground truth left flag tensor
        - labels_right_flag (torch.Tensor): ground truth right flag tensor
        - left_pred (torch.Tensor): predicted left hand tensor
        - right_pred (torch.Tensor): predicted right hand tensor
        - left_labels (torch.Tensor): ground truth left hand tensor
        - right_labels (torch.Tensor): ground truth right hand tensor
        - batch_left (torch.Tensor): batch tensor for left hand
        - batch_right (torch.Tensor): batch tensor for right hand

        Returns:
        - loss (torch.Tensor): computed loss tensor
        """
        loss_left_flag = self.criterion_left_flag(
            pred_left_flag, labels_left_flag)
        loss_right_flag = self.criterion_right_flag(
            pred_right_flag, labels_right_flag)

        loss_flags = (loss_left_flag + loss_right_flag) / 2

        # Substruct non existing
        left_pred = left_pred[batch_left != 0]
        left_labels = left_labels[batch_left != 0]
        right_pred = right_pred[batch_right != 0]
        right_labels = right_labels[batch_right != 0]

        loss_left = self.criterion_left(left_pred, left_labels)
        loss_right = self.criterion_left(right_pred, right_labels)

        loss_hands = (loss_left + loss_right)/2

        return ((0.02*loss_flags) + (0.98*loss_hands))


class FPHALoss(nn.Module):
    def __init__(self):
        super(FPHALoss, self).__init__()

        self.obj_loss = nn.CrossEntropyLoss()
        self.hand_pose_loss = IoULoss()
        # self.hand_keypoints_loss = nn.MSELoss()

    def forward(self, pred_obj, pred_pose, obj_label, labels):

        # hand_pose_loss = self.hand_pose_loss(pred_pose, labels)
        obj_loss = self.obj_loss(pred_obj, obj_label)
        # kpts_loss = self.hand_keypoints_loss(
        #     kpts_pred, kpts_gt)
        # return (0.01*obj_loss) + (0.99*hand_pose_loss)
        # return obj_loss
        # return hand_pose_loss + kpts_loss
        return obj_loss
