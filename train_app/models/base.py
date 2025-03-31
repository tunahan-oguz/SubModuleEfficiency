from __future__ import annotations

import os
import typing
from enum import Enum
from functools import partial

import cv2
import numpy as np
import pytorch_lightning as pl
import torch

from train_app.utils import copy_doc


class ModelBase(pl.LightningModule):
    """Base class for a PyTorch Lightning module.

    Parameters
    ----------
    config: dict | None
        Configuration dictionary for the model. (default: None)
    visualize: bool
        Flag indicating whether to enable visualization. (default: False)
    *args
        Additional positional arguments to be passed to the parent class.
    **kwargs
        Additional keyword arguments to be passed to the parent class.

    """

    DEFAULT_BATCH_SIZE = 4

    class State(Enum):
        """Enum class defining the possible states of the model."""

        IDLE = "idle"
        TRAINING = "train"
        VALIDATION = "validation"
        Test = "test"

        @staticmethod
        def values():
            return [item.value for item in ModelBase.State]

    def __init__(self, config: dict | None = None, visualize: bool = False, postprocess=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.visualize = visualize
        self.postprocess = postprocess
        self.state = ModelBase.State.IDLE
        batch_size = self.config["hypes"]["batch_size"] if self.config is not None else ModelBase.DEFAULT_BATCH_SIZE
        self.log = partial(self.log, batch_size=batch_size)  # type: typing.Callable

        self.save_hyperparameters(ignore=["state", "losses", "visualize", "hypes_config", "loss_config"])

    # --- private functions ---
    # WARN: in python <= 3.8, using decorators makes mem leak in GPU
    def __measure_model(self, prediction, target, *args, **kwargs):
        """Calculates the loss and metrics for the given prediction and target tensors.

        Parameters
        ----------
        prediction
            The predicted output tensor.
        target
            The target tensor.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The total loss value.

        """
        total_loss = None

        for loss, loss_config in self.losses + self.metrics:
            if self.state.value.casefold() in loss_config["apply_on"]:
                try:
                    loss_value = loss(prediction, target, *args, **kwargs)
                except TypeError:
                    loss_value = loss(prediction, target)
                loss_value *= loss_config["weights"]

                self.log(self.log_prefix + loss_config["type"], loss_value.item(), sync_dist=True)

                if (loss, loss_config) in self.losses:
                    total_loss = loss_value if total_loss is None else total_loss + loss_value

        if total_loss is None:
            raise RuntimeError("No loss calculated!")

        self.log(self.log_prefix + "loss", total_loss.item(), sync_dist=True)

        return total_loss

    def __loss_callback(func: typing.Callable):
        """Decorator function to handle loss callbacks.

        Parameters
        ----------
        func: Callable
            The function to be decorated.

        Returns
        -------
        Callable
            The decorated function.

        """

        def inner(*args, **kwargs):
            func(*args)
            for loss, loss_config in args[0].losses:
                getattr(loss, func.__name__)()

        return inner

    def __log_image(self, image_name, img):
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.MLFlowLogger):
                logger.experiment.log_image(run_id=logger.run_id, artifact_file=os.path.join("images/", image_name), image=img)
            if isinstance(logger, pl.loggers.WandbLogger):
                logger.log_image(key=image_name, images=[img])
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                img = img.transpose(-1, 0, 1)
                logger.experiment.add_image(tag=image_name, img_tensor=img)

    # WARN: in python <= 3.8, using decorators makes mem leak in GPU
    def __handle_predictions(self, batch: dict):
        """Handles the predictions for a given batch.

        Parameters
        ----------
        batch: dict
            The input batch.

        Returns
        -------
        torch.Tensor
            The total loss value.

        """
        batch = self._post_processing(batch)
        loss_value = self.__measure_model(**batch)

        if self.visualize:
            self._visualize(batch)

        return loss_value

    def _on_start(self):
        """Performs initialization steps at the start of training, validation, or testing.
        Initializes the loss and measurers for the training process.

        Raises
        ------
        RuntimeError
            If no loss is found in the configuration.

        """

        # Lazy load losses,
        # if task is inference losses will not be loaded.
        from train_app.loss import loss  # noqa
        from train_app.loss import metrics  # noqa

        self.hypes_config = self.config["hypes"]
        self.losses = []

        # TODO: Add a YAML validity checker
        def create_measurers(config, keys):
            measurers = []
            for key in keys:
                if key in config.keys():
                    self.measurer_config = config[key]
                    break
                else:
                    return []
            for measurer_item in self.measurer_config:
                measurer_config = list(measurer_item.values())[0]
                class_type = measurer_config["type"]
                measurer_class = getattr(metrics, class_type) if hasattr(metrics, class_type) else getattr(loss, class_type)
                measurer_args = measurer_config["args"] if "args" in measurer_config else {}
                measurer_obj = measurer_class(
                    current_epoch_fn=lambda: self.current_epoch,
                    state_fn=lambda: self.state,
                    device_fn=lambda: self.device,
                    log_fn=self.log,
                    log_image_fn=self.__log_image,
                    log_prefix_fn=lambda: self.log_prefix,
                    **measurer_args,
                )
                measurers.append((measurer_obj, measurer_config))
            return measurers

        self.metrics = create_measurers(self.config, ["metrics", "metric"])
        self.losses = create_measurers(self.config, ["losses", "loss"])

        if len(self.losses) == 0:
            raise RuntimeError("No loss found in config. Please define loss in the config file!")

    # --- protected functions ---
    def _post_processing(self, batch: dict):
        """You can override this method to add additional post processing capabilities.

        Parameters
        ----------
        batch: dict
            The input batch.

        Returns
        -------
        dict
            Postprocessed batch dictionary.

        """
        return batch

    def _visualize(self, batch):
        raise NotImplementedError(
            "You are trying to visualize model predictions, "
            "but the _visualize function is not implemented. See the documentation to find out how to do it."
        )

    # --- callback functions ---
    @__loss_callback
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.log_prefix = "validation/"
        self.state = ModelBase.State.VALIDATION

    @__loss_callback
    def on_test_start(self):
        super().on_test_start()
        self.log_prefix = "test/"
        self.state = ModelBase.State.Test

        self._on_start()

    @__loss_callback
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.log_prefix = "train/"
        self.state = ModelBase.State.TRAINING
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], sync_dist=True)

    @__loss_callback
    def on_train_epoch_end(self) -> None:
        self.log_prefix = "train/"
        for loss, loss_config in self.metrics:
            if hasattr(loss, "on_epoch_end"):
                loss.on_epoch_end()

    @__loss_callback
    def on_validation_epoch_end(self) -> None:
        self.log_prefix = "validation/"
        for loss, loss_config in self.metrics:
            if hasattr(loss, "on_epoch_end"):
                loss.on_epoch_end()

    @__loss_callback
    def on_test_epoch_end(self) -> None:
        self.log_prefix = "test/"
        for loss, loss_config in self.metrics:
            if hasattr(loss, "on_epoch_end"):
                loss.on_epoch_end()

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if not self.config:
            raise RuntimeError("no config found!")

        self._on_start()

    def on_validation_start(self) -> None:
        self._on_start()
        return super().on_validation_start()

    def on_save_checkpoint(self, checkpoint: dict[str, typing.Any]) -> None:
        """Callback function called when saving a checkpoint.

        Parameters
        ----
        checkpoint: dict
            The checkpoint to be saved.

        """
        if type(self.logger) == pl.loggers.mlflow.MLFlowLogger:
            epoch = checkpoint["epoch"]
            step = checkpoint["global_step"]
            run_id = self.logger.run_id
            self.logger.experiment.log_text(run_id, "", f"model/checkpoints/epoch={epoch}-step={step}/MLmodel")  # empty file
        return super().on_save_checkpoint(checkpoint)

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler for training.

        Returns
        -------
        dict
            The optimizer, learning rate scheduler, and monitoring metric.

        """
        if ("optimizers" not in self.config) or (len(self.config["optimizers"]) == 0):
            raise RuntimeError("You have to specify an optimizer under the optim tab.")
        # TODO: Support more than one optimizer
        optimizer_name = list(self.config["optimizers"].keys())[0]
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer_args = self.config["optimizers"][optimizer_name].get("args", {})
        optimizer_args["lr"] = optimizer_args.get("lr", self.config["hypes"]["lr"])
        # TODO: Add logger and use in such cases.
        if pl.utilities.rank_zero_only.rank == 0:
            print(f"Optimizer's learning rate has been set to {optimizer_args['lr']}")

        optimizer = optimizer_class(params=self.parameters(), **optimizer_args)
        scheduler = monitor = None
        if "scheduler" in self.config["optimizers"][optimizer_name]:
            scheduler_info = self.config["optimizers"][optimizer_name]["scheduler"]
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_info["type"])
            scheduler_args = scheduler_info.get("args", {})
            scheduler = scheduler_class(optimizer=optimizer, **scheduler_args)
            # TODO: Metrics should be able to detected automatically, if selected metric is not exists, warn user and use training/loss
            monitor = scheduler_info.get("monitor", "training/loss")
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
        else:
            return optimizer

    # WARN: in python <= 3.8, using decorators makes mem leak in GPU
    def training_step(self, batch: dict, batch_idx):
        batch = self.feed_batch(batch)
        loss_value = self.__handle_predictions(batch)

        return loss_value

    # WARN: in python <= 3.8, using decorators makes mem leak in GPU
    def validation_step(self, batch, batch_idx):
        batch = self.feed_batch(batch)
        loss_value = self.__handle_predictions(batch)

        return loss_value

    def test_step(self, batch, batch_idx):
        batch = self.feed_batch(batch)
        loss_value = self.__handle_predictions(batch)

        return loss_value

    # --- static functions ---
    @staticmethod
    def correct_config(config):
        """Corrects the configuration dictionary for losses and metrics. This function corrects the keys and values in the configuration dictionary
        to ensure consistent and valid definitions for losses and metrics. It modifies the input `config` dictionary in-place.

        Parameters
        ----------
        config: dict
            The configuration dictionary.

        Raises
        ------
        RuntimeError
            If an unsupported `apply_on` value or `weights` value is encountered.

        """

        def correct_loss_and_metrics(config, keys):
            if keys[0] not in config:
                if keys[1] not in config:
                    return

                tmp_loss = {keys[1]: config.pop(keys[1])}
                config[keys[0]] = [tmp_loss]

            for item in config[keys[0]]:
                cfg = item[keys[1]]

                if "apply_on" not in cfg:
                    cfg["apply_on"] = ["train", "validation", "test"]
                elif type(cfg["apply_on"]) is list:
                    if not all(x in ModelBase.State.values() for x in cfg["apply_on"]):
                        raise RuntimeError(
                            f"un supported apply_on in loss {cfg['type']}. it must be 'train', 'validation' or ['train', 'validation']"
                        )
                elif type(cfg["apply_on"]) is str:
                    if cfg["apply_on"] not in ModelBase.State.values():
                        raise RuntimeError(
                            f"un supported apply_on in loss {cfg['type']}. it must be 'train', 'validation' or ['train', 'validation']"
                        )
                else:
                    raise RuntimeError(f"un supported apply_on in loss {cfg['type']}. it must be 'train', 'validation' or ['train', 'validation']")

                if "weights" not in cfg:
                    cfg["weights"] = 1
                elif type(cfg["weights"]) is str:
                    cfg["weights"] = float(cfg["weights"])
                elif not (type(cfg["weights"]) is float or type(cfg["weights"]) is int):
                    raise RuntimeError(f"un supported weights in loss {cfg['type']}. it must be float or int")

        correct_loss_and_metrics(config, ["losses", "loss"])
        correct_loss_and_metrics(config, ["metrics", "metric"])


class MultiInputSegmetationAdapter(ModelBase):
    """Adapter class for multi-input segmentation models. This class extends the `ModelBase` class and provides additional functionality
    for multi-input segmentation models. It includes methods for visualization and processing of input batches.

    """

    def _on_start(self, *args, **kwargs):
        super()._on_start(*args, **kwargs)
        if self.visualize:
            cv2.namedWindow("loss", cv2.WINDOW_GUI_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)
            cv2.namedWindow("prediction", cv2.WINDOW_GUI_NORMAL)

    def __tensor_to_img(self, tensor_img: torch.Tensor):
        return (tensor_img * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    def _visualize(self, batch):
        """Visualize the input batch. This method displays the input images, ground truth masks, predicted masks,
        intersection image, and loss image for visualization purposes.

        Parameters
        ----------
        batch: dict
            The input batch dictionary containing "inputs", "target", and "prediction".

        """
        mask_gt = self.__tensor_to_img(batch["target"][0])
        mask_pred = self.__tensor_to_img(torch.sigmoid(batch["prediction"][0]))
        input_width = batch["inputs"][0][0].shape[2]
        input_debug = np.zeros(
            (
                batch["inputs"][0][0].shape[1],
                input_width * len(batch["inputs"]),
                batch["inputs"][0][0].shape[0],
            ),
            dtype=np.uint8,
        )
        for img_id, input_img in enumerate(batch["inputs"]):
            width_start, width_end = input_width * img_id, input_width * (img_id + 1)
            input_debug[:, width_start:width_end, :] = self.__tensor_to_img(input_img[0])

        intersection_image = cv2.bitwise_and(mask_gt, mask_pred)
        loss_image = cv2.merge((intersection_image, mask_gt, mask_pred))

        cv2.imshow("prediction", mask_pred)
        cv2.imshow("input", input_debug)
        cv2.imshow("loss", loss_image)
        cv2.waitKey(3)

    def feed_batch(self, batch):
        """Feed a batch of inputs through the model and process the output. This method takes a batch of input data, passes it through the model,
        and processes the output. It modifies the input batch dictionary by adding the "prediction" and "target" fields.

        Parameters
        ----------
        batch: dict
            The input batch dictionary.

        Returns
        -------
        dict
            The modified batch dictionary with "prediction" and "target" fields.

        """
        batch["prediction"] = self(batch["inputs"])
        batch["target"] = batch.pop("mask")
        return batch


class SemanticSegmentationAdapter(ModelBase):
    def _on_start(self, *args, **kwargs):
        super()._on_start(*args, **kwargs)
        if self.visualize:
            cv2.namedWindow("target", cv2.WINDOW_GUI_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)
            cv2.namedWindow("prediction", cv2.WINDOW_GUI_NORMAL)

    def __tensor_to_img(self, tensor_img: torch.Tensor):
        if tensor_img.size(0) == 3:
            tensor_img = tensor_img.permute(1, 2, 0)
        return tensor_img.detach().cpu().numpy().astype(np.uint8)

    def _visualize(self, batch):
        mask_gt = self.__tensor_to_img(batch["target"][0] * (batch["target"][0] >= 0))
        mask_pred = self.__tensor_to_img(torch.softmax(batch["prediction"][0], dim=0).argmax(dim=0))

        if self.postprocess:
            mask_pred = self.postprocess(mask_pred)
            mask_gt = self.postprocess(mask_gt)

        input = self.__tensor_to_img(batch["inputs"][0] * 255)

        cv2.imshow("prediction", mask_pred)
        cv2.imshow("input", input)
        cv2.imshow("target", mask_gt)
        cv2.waitKey(1)

    @copy_doc(MultiInputSegmetationAdapter.feed_batch)
    def feed_batch(self, batch):
        batch["prediction"] = self(batch["inputs"])
        batch["target"] = batch.pop("mask")
        return batch


class ClassificationAdapter(ModelBase):
    @copy_doc(MultiInputSegmetationAdapter.feed_batch)
    def feed_batch(self, batch):
        batch["prediction"] = self(batch["inputs"])
        batch["target"] = batch.pop("labels")

        return batch
