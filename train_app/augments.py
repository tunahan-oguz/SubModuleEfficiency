from __future__ import annotations

import random


class BaseCompose:
    """Base class for composing and preprocessing data for transformations.
    This class provides methods for checking the validity of data, preprocessing the data
    to a standardized format, and postprocessing the preprocessed data to original format. It supports various data
    types such as images, masks, bounding boxes, keypoints, and class labels.

    Attributes
    ----------
    default_params: set
        Set of default parameter names for data types.
    """

    default_params = {"images", "masks", "bboxes", "keypoints", "class_labels"}

    def _check_data_validity(self, **data):
        """Checks the validity of the input data.

        Parameters
        ----------
        **data
            Keyword arguments representing different data types.

        Raises
        ------
        KeyError
            If the required data types are not present or if there are inconsistencies in the data.

        """
        if "images" not in data:
            raise KeyError("Please make sure images that will be transformed are sended with 'images' keyword.")
        if ("bboxes" in data or "keypoints" in data) and "class_labels" not in data:
            raise KeyError("Bboxes/keypoints must have class labels.")
        if "bboxes" in data and len(data["class_labels"]) != len(data["bboxes"]):
            raise KeyError("All bboxes must have one class labels.")
        if "keypoints" in data and len(data["class_labels"]) != len(data["keypoints"]):
            raise KeyError("All keypoints must have one class labels.")

    def _preprocess_data(self, **data):
        """Preprocesses the input data into a standardized format.

        Parameters
        ----------
        **data
            Keyword arguments representing different data types.

        Returns
        -------
        dict
            Preprocessed data in a standardized format.

        """
        preprocessed_data = {}
        preprocessed_data["image"] = data["images"][0]
        preprocessed_data["mask"] = data["masks"][0] if "masks" in data else None
        preprocessed_data["bboxes"] = data["bboxes"] if "bboxes" in data else []
        preprocessed_data["keypoints"] = data["keypoints"] if "keypoints" in data else []
        preprocessed_data["class_labels"] = data["class_labels"] if "class_labels" in data else []

        # If there is more than images or masks distribute them to different keys (like images, images1, images2)
        if "masks" in data.keys():
            preprocessed_data.update({f"masks{i+1}": data["masks"][i + 1] for i in range(len(data["masks"]) - 1)})
        preprocessed_data.update({f"image{i+1}": data["images"][i + 1] for i in range(len(data["images"]) - 1)})

        # Add additional arguments
        preprocessed_data.update({key: data[key] for key in set(data.keys()) - self.default_params})

        # Remove None arguments
        for key in list(preprocessed_data.keys()):
            if preprocessed_data[key] is None:
                del preprocessed_data[key]

        return preprocessed_data

    def _postprocess_data(self, **data):
        """Postprocesses the input data by extracting specific data types and organizing them into a dictionary.

        Parameters
        ----------
        **data
            Keyword arguments representing different data types.

        Returns
        -------
        dict
            Postprocessed data organized in a dictionary format.

        """
        postprocessed_data = {}
        postprocessed_data["images"] = [item for key, item in data.items() if "image" in key]

        if "mask" in data:
            postprocessed_data["masks"] = [item for key, item in data.items() if "mask" in key]
        if "bboxes" in data:
            postprocessed_data["bboxes"] = data["bboxes"]
        if "keypoints" in data:
            postprocessed_data["keypoints"] = data["keypoints"]
        if "class_labels" in data:
            postprocessed_data["class_labels"] = data["class_labels"]

        return postprocessed_data


class Compose(BaseCompose):
    """Composes several transforms together.

    Parameters
    ----------
    transforms: list
        list of transformations to compose
    p: float
        probability of applying all list of transforms. Default 1.0.
    """

    def __init__(
        self,
        transforms: list,
        p: float = 1.0,
    ):
        self.transforms = transforms
        self.p = p

    def __call__(self, *args, **data):
        """Applies the composed augmentations to the input data.

        Parameters
        ----------
        *args
            Positional arguments (not used).
        **data
            Keyword arguments representing different data types.

        Returns
        -------
        dict
            Augmented data organized in a dictionary format.

        Raises
        ------
        KeyError
            If data is not passed as named arguments.

        """

        if args:
            raise KeyError("You have to pass data to augmentations as named arguments.")

        self._check_data_validity(**data)

        data = self._preprocess_data(**data)

        if random.random() <= self.p:
            for transform in self.transforms:
                data = transform(**data)

        data = self._postprocess_data(**data)

        return data
