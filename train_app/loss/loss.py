from __future__ import annotations

from typing import Callable

import torch


class BaseLoss(torch.nn.Module):
    """The BaseLoss class is designed to be inherited from and extended to create custom loss functions tailored to specific needs.
    By subclassing BaseLoss, you can define your own loss functions with access to important attributes such as the current epoch,
    the state of the model, and the device on which the model is running. This allows you to customize the behavior during different
    training or evaluation phases.

    Parameters
    ----------
    current_epoch_fn: Callable
        A function that returns the current epoch.
    state_fn: Callable
        A function that returns the current state of the model.
    device_fn: Callable
        A function that returns the device on which the model is running.
    log_fn: Callable
        A function used for logging messages.
    log_image_fn: Callable
        A function used for logging images.
    log_prefix_fn: Callable
        A function that returns a prefix for log messages.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    current_epoch: int
        The current epoch of the training process.
    state: object
        The current state of the model.
    device: torch.device
        The device on which the model is running.
    log: Callable
        A function used for logging messages.
    log_image: Callable
        A function used for logging images.
    log_prefix: str
        A prefix for log messages.

    """

    def __init__(
        self,
        current_epoch_fn: Callable,
        state_fn: Callable,
        device_fn: Callable,
        log_fn: Callable,
        log_image_fn: Callable,
        log_prefix_fn: Callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.current_epoch_fn = current_epoch_fn
        self.state_fn = state_fn
        self.device_fn = device_fn
        self.log_fn = log_fn
        self.log_image_fn = log_image_fn
        self.log_prefix_fn = log_prefix_fn

    @property
    def current_epoch(self):
        # return self.__model_base.current_epoch
        return self.current_epoch_fn()

    @property
    def state(self):
        return self.state_fn()

    @property
    def device(self):
        return self.device_fn()

    @property
    def log(self):
        return self.log_fn

    @property
    def log_image(self):
        return self.log_image_fn

    @property
    def log_prefix(self):
        return self.log_prefix_fn()

    def on_train_epoch_start(self):
        """Hook method called at the start of each training epoch."""
        pass

    def on_validation_epoch_start(self):
        """Hook method called at the start of each validation epoch."""
        pass

    def on_test_start(self):
        """Hook method called at the start of the testing phase."""
        pass

    def on_train_epoch_end(self):
        """Hook method called at the end of each training epoch."""
        pass

    def on_validation_epoch_end(self):
        """Hook method called at the end of each validation epoch."""
        pass

    def on_test_end(self):
        """Hook method called at the end of the testing phase."""
        pass
    
    def on_test_epoch_end(self):
        """Hook method called at the end of test epoch."""
        pass

class CrossEntropyLoss(BaseLoss):
    """Cross entropy loss for classification tasks.
    This class extends the BaseLoss class and implements the forward method for computing
    the cross entropy loss between model predictions and target labels.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    crossEntropyLoss: torch.nn.CrossEntropyLoss
        Instance of the CrossEntropyLoss class.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.

    """

    def __init__(self, ignore_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, prediction, target):
        """Computes the cross entropy loss between prediction and target."""
        loss = self.crossEntropyLoss(prediction, target)
        # accuracy = torch.sum(torch.argmax(inputs, dim=1) == targets) / inputs.shape[0]
        return loss

class UncertaintyAwareCE(BaseLoss):

    def __init__(self, ignore_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, prediction, target):
        
        m5, m4, m3, m2, m1 = prediction
        loss_m5 = self.crossEntropyLoss(m5, target)
        loss_m4 = self.crossEntropyLoss(m4, target)
        loss_m3 = self.crossEntropyLoss(m3, target)
        loss_m2 = self.crossEntropyLoss(m2, target)
        loss_m1 = self.crossEntropyLoss(m1, target)
        loss = loss_m1 + loss_m2 + loss_m3 + loss_m4 + loss_m5
        # self.log(self.log_prefix + "CELossm5", loss_m5.item())
        # self.log(self.log_prefix + "CELossm4", loss_m4.item())
        # self.log(self.log_prefix + "CELossm3", loss_m3.item())
        # self.log(self.log_prefix + "CELossm2", loss_m2.item())
        # self.log(self.log_prefix + "CELossm1", loss_m1.item())

        return loss