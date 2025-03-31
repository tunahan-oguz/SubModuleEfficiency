from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class BaseMetric:
    """The BaseMetric class is designed to be inherited from and extended to create custom metric functions."""

    def __init__(
        self,
        device_fn: Callable,
        log_fn: Callable,
        log_image_fn: Callable,
        log_prefix_fn: Callable,
        *args,
        **kwargs,
    ):
        self.device_fn = device_fn
        self.log = log_fn
        self.log_image = log_image_fn
        self.log_prefix_fn = log_prefix_fn

    @property
    def log_prefix(self):
        return self.log_prefix_fn()

    @log_prefix.setter
    def log_prefix(self, value):
        self.log_prefix = value

    def on_epoch_end(self):
        pass


class PixelAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        """Calculates the pixel accuracy metric for image segmentation tasks.
        This metric compares the predicted segmentation labels with the ground truth labels and measures the percentage
        of correctly classified pixels out of all labeled pixels in the input images.

        Parameters
        ----------
        predictions: torch.Tensor
            Tensor containing the predicted segmentation labels.
        labels: torch.Tensor
            Tensor containing the ground truth segmentation labels.

        Returns
        -------
        torch.Tensor
            The pixel accuracy as a scalar tensor.

        Notes
        -----
        - The input tensor `prediction` should have 4 dimensions (N,C,H,W)
        - The input tensor `labels` should have 3 dimension (N,H,W)
        - The predicted labels are obtained by taking the argmax along the second dimension of `predictions`.
        - Both `predictions` and `labels` should have integer values representing class labels.
        - Unlabeled pixels (where the value in `labels` is greater than the maximum class label) are ignored in the calculation.

        """
        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        predict = torch.argmax(predictions, 1)
        pixel_labeled = torch.sum(labels >= 0)
        pixel_correct = torch.sum((predict == labels) * (labels >= 0))
        pixel_accuracy = pixel_correct / (pixel_labeled + torch.finfo(torch.float32).eps)

        return pixel_accuracy


class IoU(BaseMetric):
    def __init__(self, K, smooth=1e-7, ignore_index= -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.K = K
        self.smooth = smooth

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        """
        Calculates the IoU score given the predicted and ground truth labels.

        Parameters
        ----------
        predictions: torch.Tensor
            Tensor containing predicted labels.
        labels: torch.Tensor
            Tensor containing ground truth labels.

        Returns
        -------
        torch.Tensor
            mean Intersection over Union (mIoU) score.

        Notes
        -----
        - The input tensor `prediction` should have 4 dimensions (N,C,H,W)
        - The input tensor `labels` should have 3 dimension (N,H,W)
        - The predicted labels are expected to be in the form of logits or probabilities, where the channel dimension
        represents the number of classes.
        - The ground truth labels should be integers indicating the class indices.
        - The IoU score is computed as the average IoU over all classes.
        - This implementation handles cases where the number of unique labels in the ground truth is less than the
        number of classes in the predictions. Labels beyond the unique labels are ignored in the calculations.
        - The IoU score is calculated using the intersection and union of predicted and ground truth labels.

        """

        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
                
        if self.K == 2:
            intersection = output * target
            union = torch.clamp(output + target, max=1)
            return (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
        
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection.float(), bins=self.K, min=0, max=self.K-1)
        area_output = torch.histc(output.float(), bins=self.K, min=0, max=self.K-1)
        area_target = torch.histc(target.float(), bins=self.K, min=0, max=self.K-1)
        area_union = area_output + area_target - area_intersection
        return (area_intersection.sum() + self.smooth) / (area_union.sum() + self.smooth)

class mIoU(BaseMetric):
    def __init__(self, K, smooth=1e-7, ignore_index= -1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.K = K
        self.smooth = smooth

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
                
        if self.K == 2:
            intersection = output * target
            union = torch.clamp(output + target, max=1)
            return (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
        
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection.float(), bins=self.K, min=0, max=self.K-1)
        area_output = torch.histc(output.float(), bins=self.K, min=0, max=self.K-1)
        area_target = torch.histc(target.float(), bins=self.K, min=0, max=self.K-1)
        area_union = area_output + area_target - area_intersection
        return ((area_intersection) / (area_union)).nanmean()

class AP(BaseMetric):
    def __init__(self, K, smooth=1e-7, ignore_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.K = K
        self.smooth = smooth

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        return precision_score(target, output, labels=range(self.K), average="weighted") 

class AR(BaseMetric):
    def __init__(self, K, smooth=1e-7, ignore_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.K = K
        self.smooth = smooth

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        return recall_score(target, output, labels=range(self.K), average="weighted")

class AF1(BaseMetric):
    def __init__(self, K, smooth=1e-7, ignore_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.K = K
        self.smooth = smooth

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        if isinstance(predictions, (tuple, list)): predictions = predictions[-1]

        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        output[target == self.ignore_index] = self.ignore_index
        return f1_score(target, output, labels=range(self.K), average="weighted")