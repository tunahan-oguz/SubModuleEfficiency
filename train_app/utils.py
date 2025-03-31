from __future__ import annotations

import atexit
import collections.abc
import contextlib
import glob
import importlib
import inspect
import os
import re
import shutil
import time
from enum import Enum
from typing import Any
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks
import torch
import wandb
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from PIL import Image

import train_app.dataset as dataset  # noqa
from train_app.registers.model_registry import model_registry



class TerminalColors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def prepare_directory(path, project, with_weights=True):
    """Prepares an empty directory in the given directory path to store several train outputs.

    Parameters
    ----------
    path : str
        Root path to be prepared.
    project : str
        Name of the project.
    with_weights : bool, optional
        Whether to include weights directory or not, by default True

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Raised when the given path parameter is not a directory.

    """
    tmp_path = os.path.join(path, project)
    os.makedirs(tmp_path, exist_ok=True)
    n_runs = len(os.listdir(tmp_path))
    path = os.path.join(tmp_path, f"version_{n_runs}")
    os.mkdir(path)
    if with_weights:
        os.mkdir(os.path.join(path, "weights"))
    return path


def read_yaml(path):
    """read the yaml from path

    Parameters
    ----------
    path : string
        yml file path

    Returns
    -------
    dictionary
        yaml config dict

    """
    with open(path) as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)


# TODO: add logger
def generate_model(model_config):
    """Generate a model based on the provided configuration.

    Parameters
    ----------
    model_config: dict
        Configuration for the model.

    Returns
    -------
    model
        Generated model instance.
    model_class
        Model class type.

    Raises
    ------
    ImportError
        If there is an issue with importing the model class.

    """
    model_class_type = model_config["type"]

    
    model_class = model_registry.registered_models.get(model_class_type, None)
    assert model_class is not None, "The model that has been attempted to be used is not properly registered."
    model_args = model_config["args"] if "args" in model_config.keys() else {}
    model = model_class(**model_args)
    return model, model_class


def generate_dataset(dataset_conf, task_type):
    """Generates dataset classes which includes augmentation and other stuff.

    Parameters
    ----------
    dataset_conf: dict
        dictionary that contains dataset informations

    task_type: str
        the type of task(train, valid, test)
    """
    tmp_conf = dataset_conf.copy()
    if task_type not in tmp_conf:
        raise (f"error! dataset_conf not have {task_type}")

    dataset_class_type = tmp_conf[task_type]["type"]
    del tmp_conf[task_type]["type"]

    tmp_type_config = tmp_conf[task_type]
    type_keys = ["train", "valid", "validation", "test"]
    for key in type_keys:
        if key in tmp_conf:
            del tmp_conf[key]
    tmp_conf.update(tmp_type_config)

    if "torchvision" in dataset_class_type:  # Create torchvision wrapper
        dataset_class = eval(dataset_class_type)
        dataset_ = dataset_class(root="./dataset", download=True, train="train" in task_type)

        dataset_ = dataset.TorchVisionDataset(dataset=dataset_, **tmp_conf)
    else:
        import_from = dataset
        if "import_from" in tmp_conf:
            import_from = importlib.import_module(tmp_conf["import_from"])
            del tmp_conf["import_from"]

        dataset_class = getattr(import_from, dataset_class_type)
        dataset_ = dataset_class(**tmp_conf)

    return dataset_, dataset_class


def deep_update(first: dict[Any, Any], second: dict[Any, Any]) -> dict[Any, Any]:
    """deep update the first dict with second dict. If first dict has a node contains second dict, then the related part of the
    first dict gets updated only.

    Parameters
    ----------
    first : dict[Any, Any]
        dict to update on
    second : dict[Any, Any]
        dict update from

    Returns
    -------
    dict[Any, Any]
        updated firct dict

    """

    for k, v in second.items():
        if isinstance(v, collections.abc.Mapping):
            first[k] = deep_update(first.get(k, {}), dict(v))
        else:
            first[k] = v
    return first


def get_model_path(model_path: str) -> str:
    """Returns the path of model, if wandb path is given,
    downloads the model from wandb and return its path

    Parameters
    ----------
    model_path : str
        the path of model or wandb artifact path

    Returns
    -------
    str
        the path of the model in system

    Raises
    ------
    RuntimeError
        if wandb artifact not exists
    RuntimeError
        if Error in downloaded model
    RuntimeError
        if model not exists

    """
    if os.path.exists(model_path):
        return model_path

    if wandb.run is None:
        wandb.init()

    try:
        wandb_run = wandb.run
        if wandb_run is None:
            raise RuntimeError("wandb.run is None")

        artifact = wandb_run.use_artifact(model_path, type="model")
        tmp_path = f"/tmp/falcon-trainer/{str(time.time())}/"
        os.makedirs(tmp_path, exist_ok=True)
        shutil.rmtree(tmp_path)
        model_download_dir_path = artifact.download(root=tmp_path)

        def exit_handler():
            shutil.rmtree(tmp_path, ignore_errors=True)

        atexit.register(exit_handler)

        model_path_tmp = glob.glob(model_download_dir_path + "*.ckpt")
        if len(model_path_tmp) != 1:
            raise RuntimeError("There is more or less than one ckpt model in downloaded directory!")

        model_path = model_path_tmp[0]
    except wandb.errors.CommError:
        raise RuntimeError(
            f"There is no model with path: {model_path}. Check your model path or \
            is wandb logged in correst hots! sample connection: 'wandb login --host=http://10.0.70.247:8080 --relogin'"
        )

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model path not exists. Path: {model_path}. If you want to download \
            model from wandb in training server, use: 'wandb login --host=http://10.0.70.247:8080 --relogin'"
        )
    return model_path


def load_model(model_path: str, model_config: dict, force_config: bool = False):
    """Generates the model and and loads the state dict from given model path.
    Model path can be a local file or wandb artifact path

    Parameters
    ----------
    model_path : str
        local file or wandb artifact path
    model_config : dict, optional
        if checkpoint not have model config than will use this config., by default None

    Returns
    -------
    _type_
        model and model_config

    Raises
    ------
    err
        if unknown hyper_parameters in model checkpoint
    RuntimeError
        if both model config and checkpoint's model_config is none

    """
    model_path = get_model_path(model_path)
    model_dict = torch.load(model_path, map_location="cpu")

    if not force_config and "hyper_parameters" in model_dict:
        try:
            model_config_tmp = model_dict["hyper_parameters"]["config"]["model"]
            if model_config is not None and pl.utilities.rank_zero_only.rank == 0:
                print(TerminalColors.RED + "Warn! Loading model with the original config: ")
                print(yaml.dump({"model": model_config_tmp}, indent=4) + TerminalColors.END)

            model_config = model_config_tmp
        except KeyError as err:
            if model_config is None:
                raise err
    elif model_config is None:
        raise RuntimeError("Model config is none! No hyper_parameters in model.")

    model, _ = generate_model(model_config)

    if "state_dict" in model_dict:
        # corrects the keys in state_dict
        # model_dict["state_dict"] = {
        #     key.replace("model.", "").replace("unet.", ""): model_dict["state_dict"][key]
        #     for key in model_dict["state_dict"].keys()
        #     if "criterion.bce_loss.pos_weigh" not in key or "criterion1.pos_weight" not in key
        # }
        model.load_state_dict(model_dict["state_dict"])
    else:
        model.load_state_dict(model_dict)

    return model, model_config


@rank_zero_only
def print_config(config):
    """Prints the config with yaml formatted.

    Parameters
    ----------
    config : dict
        The config map of the train

    """
    if config is None:
        return

    print(TerminalColors.GREEN + " --- config begin --- \n" + TerminalColors.END)
    print(TerminalColors.CYAN + yaml.dump(config, indent=4) + TerminalColors.END)
    print(TerminalColors.GREEN + " --- config end --- " + TerminalColors.END)


def with_caller_fn(func):
    """Decorator that adds a `caller_fn` argument to the decorated function. The `caller_fn` argument represents the name of the function that called
    the decorated function. If the `caller_fn` argument is already provided in the function call, it will be used as is. Otherwise, the decorator will
    automatically determine the caller function's name using the stack trace.

    Parameters
    ----------
    func: callable
        Function to be decorated.

    Returns
    -------
    callable
        Decorated function with the added `caller_fn` argument.

    Raises
    ------
    RuntimeError
        If the decorator is unable to determine the caller function's name from the stack trace.

    """

    def ____caller_finder_function_unique_name(*args, **kwargs):
        arg_key = "caller_fn"
        if arg_key not in kwargs:
            stack_pos = None
            stacks = inspect.stack(0)
            for i, stack in enumerate(stacks):
                if stack[3] == "____caller_finder_function_unique_name":
                    stack_pos = i + 1
                    break

            if stack_pos is None:
                raise RuntimeError("not found in stack")

            kwargs[arg_key] = stacks[stack_pos][3]

        return func(*args, **kwargs)

    return ____caller_finder_function_unique_name


def mask_2_bbox(masks, threshold=127):
    """Calculates bounding box predictions from predicted masks

    Parameters
    ----------
    masks: list
        The masks as ndarray that will used to calculate bounding boxes `(batch, channel, width, height)`

    threshold: int
        Value that used to threshold masks

    Returns:
    --------

    bboxes: list
        Bounding boxes `(batch, n_bbox, 4)`

    """
    # TODO: Support multiclass bbox

    def preprocess_mask(img, threshold):
        _, img_processed = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        img_processed = cv2.dilate(img_processed, np.ones((5, 5), np.uint8), iterations=3)
        img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        return img_processed

    return [img_to_boxes(preprocess_mask(mask, threshold)) for mask in masks]


def img_to_boxes(image):
    """Extracts bounding boxes from predicted 2d masks

    Parameters
    ----------
    image: numpy.ndarray
        The input image.

    Returns
    -------
        numpy.ndarray: An array of bounding boxes in the format (x1, y1, x2, y2).

    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    # convert to (x1, y1, x2, y2) format
    bboxes = np.array([[rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]] for rect in bboxes]).reshape(len(bboxes), 4)
    return NMS(bboxes)


def NMS(boxes, overlap_thresh=0.4):
    """Applies non-maximum suppression (NMS) to a set of bounding boxes.

    Parameters
    ----------
    boxes: numpy.ndarray
        An array of bounding boxes in the format (x1, y1, x2, y2). overlapThresh (float, optional): The overlap threshold for
        suppressing overlapping boxes. Default is 0.4.

    overlap_thresh: float
        Threshold to suppress bboxes.

    Returns
    -------
    numpy.ndarray
        An array of non-overlapping bounding boxes.

    Note
    ----
    This function removes overlapping bounding boxes based on the overlap threshold. Only the boxes
    with the highest confidence or priority are retained, while the rest are suppressed.

    """
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=float)
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices, 0])
        yy1 = np.maximum(box[1], boxes[temp_indices, 1])
        xx2 = np.minimum(box[2], boxes[temp_indices, 2])
        yy2 = np.minimum(box[3], boxes[temp_indices, 3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual bounding box has an overlap bigger than threshold with any other box, remove it's index
        if np.any(overlap) > overlap_thresh:
            indices = indices[indices != i]
    # return only the boxes at the remaining indices
    return boxes[indices].astype(int)


def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    boxA: tuple
        The coordinates of the first bounding box in the format (x1, y1, x2, y2).
    boxB: tuple
        The coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns
    -------
    float
        The IoU value.

    Note
    ----
    Bounding boxes are represented as (x1, y1, x2, y2), where (x1, y1) are the top-left coordinates
    and (x2, y2) are the bottom-right coordinates.

    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calculate_precision_recall(bbox_true, bbox_pred, iou_threshold):
    """Calculates precision and recall values based on the predicted and true bounding boxes.

    Parameters
    ----------
    bbox_true: list
        List of true bounding boxes.
    bbox_pred: list
        List of predicted bounding boxes.
    iou_threshold: float
        The IoU threshold for considering a match.

    Returns
    -------
    tuple
        A tuple containing precision and recall values.

    Notes
    -----
    Precision is the ratio of true positives to the sum of true positives and false positives.
    Recall is the ratio of true positives to the sum of true positives and false negatives.
    Bounding boxes are represented as (x1, y1, x2, y2), where (x1, y1) are the top-left coordinates
    and (x2, y2) are the bottom-right coordinates.

    """
    true_positives = 0
    false_positives = 0

    bbox_pred, bbox_true = np.asarray(bbox_pred), np.asarray(bbox_true)
    bbox_pred[bbox_pred < 0] = 0
    bbox_true[bbox_true < 0] = 0

    # Calculate IoU for each predicted bounding box
    for bbox_p in bbox_pred:
        found_match = False
        for bbox_t in bbox_true:
            iou = calculate_iou(bbox_p, bbox_t)
            if iou >= iou_threshold:
                true_positives += 1
                found_match = True
                break
        if not found_match:
            false_positives += 1

    false_negatives = len(bbox_true) - true_positives

    precision = true_positives / (true_positives + false_positives) if len(bbox_pred) != 0 else -1
    recall = true_positives / (true_positives + false_negatives) if len(bbox_true) != 0 else -1

    return precision, recall


def copy_doc(copy_func: Callable) -> Callable:
    """Copies docstring of given callable

    Parameters
    ----------
    copy_func: Callable

    Example
    -------
    >>> copy_doc(self.copy_func)(self.func) or used as decorator

    """

    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapper


def generate_callbacks(callbackconfig):
    if callbackconfig is None:
        return []

    def new_callback(class_name, *args, **kwargs):
        callback = getattr(pytorch_lightning.callbacks, class_name)
        return callback(*args, **kwargs)

    return [new_callback(callback, **callbackconfig[callback]) for callback in callbackconfig.keys()]


class Profiler:
    class ProfilerState(Enum):
        ACTIVE = "active"
        PASSIVE = "passive"

    state: ProfilerState = ProfilerState.PASSIVE
    log_every_n_epochs = 2
    log_initial_epoch = False

    @staticmethod
    def load_cfg(cfg):
        Profiler.log_every_n_epochs = cfg["log_every_n_epochs"]
        Profiler.log_initial_epoch = cfg["log_initial_epoch"]

        if Profiler.log_every_n_epochs == -1:
            Profiler.logging = False

    @staticmethod
    @contextlib.contextmanager
    def profile_time(logger_func, log_name, log_cpu_time=True, log_gpu_time=True, current_epoch=None):
        """Provides synchronized time profiling for both gpu and cpu"""

        if current_epoch is not None:
            if Profiler.log_every_n_epochs == -1:
                Profiler.state = Profiler.ProfilerState.PASSIVE
            elif current_epoch % Profiler.log_every_n_epochs == 0:
                Profiler.state = Profiler.ProfilerState.ACTIVE

            if Profiler.log_initial_epoch and current_epoch == 0:
                Profiler.state = Profiler.ProfilerState.ACTIVE
        else:
            Profiler.state = Profiler.ProfilerState.ACTIVE

        if log_gpu_time and Profiler.state == Profiler.ProfilerState.ACTIVE:
            stream = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream.record_event(start)

        try:
            cpu_start = time.monotonic()
            yield
        finally:
            cpu_end = time.monotonic()
            cpu_time = (cpu_end - cpu_start) * 1000

            if log_gpu_time and Profiler.state == Profiler.ProfilerState.ACTIVE:
                stream.record_event(end)
                end.synchronize()
                gpu_time = start.elapsed_time(end)
                logger_func("profiler/" + log_name + "_gpu", gpu_time)

            if log_cpu_time and Profiler.state == Profiler.ProfilerState.ACTIVE:
                logger_func("profiler/" + log_name + "_cpu", cpu_time)

def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def pil_loader(data_path, label_path):
    """Loads a sample and label image given their path as PIL images.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.

    Returns the image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label


def remap(image, old_values, new_values):
    assert isinstance(image, Image.Image) or isinstance(
        image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(
        old_values), "new_values and old_values must have the same length"

    # If image is a PIL.Image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Replace old values by the new ones
    tmp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            tmp[image == old] = new

    return Image.fromarray(tmp)


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq

def get_entropy(logits):
    # logits of shape B x C X W x H
    p = torch.softmax(logits, dim=1)
    log_p = torch.log(p)
    return torch.sum(-1 * log_p * p, dim=1)


def tensor_to_heatmap(tensor):
    # Apply softmax to the tensor along the channel dimension
    softmaxed_tensor = torch.softmax(tensor, dim=1)
    
    # Find the maximum probability per pixel (max along the channel dimension)
    max_probs, _ = torch.max(softmaxed_tensor, dim=1)
    
    # Convert to numpy for visualization
    max_probs_np = max_probs.cpu().detach().numpy()
    
    # Normalize the max_probs to the range [0, 1] for better visualization
    min_val = np.min(max_probs_np)
    max_val = np.max(max_probs_np)
    normalized_probs = (max_probs_np - min_val) / (max_val - min_val)
    
    img = np.reshape(normalized_probs, normalized_probs.shape[-2:])
    heatmap = plt.get_cmap('turbo')(img)[:, :, :3]  # Get the RGB values from the colormap
    heatmap = (heatmap * 255).astype(np.uint8)  # Convert to 8-bit image
    
    return heatmap