from __future__ import annotations

import os
import random
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

import train_app.augments as augments
import train_app.utils as utils


tfms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomErasing(p=0.5),
        # transforms.RandomCrop(size=(64, 64))
        # transforms.RandomAffine(degrees=(-5, 5), translate=(0.5, 0.5), scale=(1, 2))
        # transforms.RandomRotation(degrees=(-30, 30)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        # transforms.Normalize((0.5,), (0.5,)),  # imagenet
    ]
)


class Dataset(torch.utils.data.Dataset):
    to_tensor = torchvision.transforms.ToTensor()

    def __init__(self, *args, **kwargs):
        self.composed_transforms = None
        self.kwargs = kwargs

    def setup_augmentations(self, aug_config):
        transforms = []
        for class_name, params in aug_config.items():
            params = {} if params is None else params

            if "A" == class_name[0] and hasattr(A, class_name[2:]):
                transforms.append(self.prepare_albumentation_augmentor(class_name[2:], params, self.n_images, self.n_masks))
            elif hasattr(augments, class_name):
                transforms.append(getattr(augments, class_name)(**params))
            else:
                raise Exception(f":( Sorry there is no releated augmentation function called '{class_name}'...")

        composed_transforms = augments.Compose(transforms)

        self.composed_transforms = composed_transforms

    def prepare_albumentation_augmentor(self, class_name, params, n_images, n_masks):
        augmentation_func = getattr(A, class_name)(**params)

        additional_targets = {}
        if n_images > 1 or n_masks > 1:
            for idx in range(n_images - 1):
                additional_targets[f"image{idx+1}"] = "image"
            for idx in range(n_masks - 1):
                additional_targets[f"mask{idx+1}"] = "mask"

        transfom = A.Compose(
            [augmentation_func], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]), additional_targets=additional_targets
        )

        return transfom

    def apply_augmentations(self, *args, **data):
        # Lazy load transforms
        if self.composed_transforms is None:
            self.n_images = len(data["images"])
            self.n_masks = len(data["masks"]) if "masks" in data else 0
            if "augment" in self.kwargs:
                self.setup_augmentations(self.kwargs["augment"])

        # To Do: if self.composed_transforms is still None that means there is no any augmentations warn the user with logger

        return self.composed_transforms(*args, **data) if self.composed_transforms is not None else data

    def check_dataset_validity(self):
        """
        Checks dataset validity with certain assertions that are specific to dataset format.
        You must override this method if you want to check your dataset validity.

        Raise:
            - NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError("This method must be overriden according to your dataset.")

    @staticmethod
    def images_to_tensors(imgs):
        tensor = []
        for img in imgs:
            if torch.is_tensor(img):
                tensor.append(img)
            else:
                tensor.append(Dataset.to_tensor(img))
        return tensor


class TorchVisionDataset(Dataset):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.asarray(img, dtype=np.uint8)  # .transpose(-1, 0, 1)

        transformed = self.apply_augmentations(images=[img])
        images = self.images_to_tensors(transformed["images"])[0]

        return {"inputs": images, "labels": label}
    
    
class RescueNet(Dataset):
    """RescueNet-v2.0 dataset: ....

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = "train/train-org-img/"
    train_lbl_folder = "train/train-label-img/"

    # Validation dataset root folders
    val_folder = "val/val-org-img/"
    val_lbl_folder = "val/val-label-img/"

    # Test dataset root folders
    test_folder = "test/test-org-img/"
    test_lbl_folder = "test/test-label-img/"

    # Filters to find the images
    org_img_extension = '.jpg'
    #lbl_name_filter = '.png'

    lbl_img_extension = '.png'
    lbl_name_filter = 'lab'

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    # The values above are remapped to the following

    new_classes =  (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0)
    # new_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('water', (61, 230, 250)),
            ('building-no-damage', (180, 120, 120)),
            ('building-medium-damage', (235, 255, 7)),
            ('building-major-damage', (255, 184, 6)),
            ('building-total-destruction', (255, 0, 0)),
            ('vehicle', (255, 0, 245)),
            ('road-clear', (140, 140, 140)),
            ('road-blocked', (160, 150, 20)),
            ('tree', (4, 250, 7)),
            ('pool', (255, 235, 0))
    ])


    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 n_classes=11,
                 img_sz=(720, 720),
                 loader=utils.pil_loader,
                 image_processor=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import SegformerImageProcessor
        self.image_processor = None
        if image_processor:
            self.image_processor = SegformerImageProcessor(do_reduce_labels=True, size={"height": img_sz[0], "width": img_sz[1]})

        self.root_dir = root_dir
        self.mode = mode
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_sz)])
        self.transform = transform
        if label_transform is None:
            # label_transform = transforms.Compose([transforms.Lambda(lambda x : x.astype(np.float32)), transforms.ToTensor()])
            label_transform = transforms.Compose([transforms.PILToTensor(), 
                                                  transforms.Resize(img_sz, interpolation=torchvision.transforms.InterpolationMode.NEAREST), 
                                                  transforms.Lambda(lambda x : x.long().squeeze())])
        self.label_transform = label_transform
        self.loader = loader
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.new_classes = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0) if n_classes == 8 else self.full_classes
        self.new_classes = (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0) if n_classes == 2 else self.new_classes
        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.org_img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.org_img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.org_img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        elif self.mode.lower() == 'vis':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.org_img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def _normalize(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
        
    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        
        if self.mode == 'vis':
            img = Image.open(self.test_data[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return self.normalize(img), os.path.basename(self.test_data[index])
        
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        
        # Remap class labels
        label = utils.remap(label, self.full_classes, self.new_classes)

        # SegFormer has its own image processor use it.
        if self.image_processor:
            encodings = self.image_processor(img, label, return_tensors="pt")
            {"inputs": encodings["pixel_values"], "mask": encodings['labels'], "path": data_path}

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return {"inputs": self.normalize(img), "mask": label, "path": data_path}

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        elif self.mode.lower() == 'vis':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
 

class InriaDataset(Dataset):
    CLASSES = ('Building', 'Background')
    PALETTE = [[255, 255, 255],  [0, 0, 0]]
    ORIGIN_IMG_SIZE = (512, 512)
    INPUT_IMG_SIZE = (512, 512)
    TEST_IMG_SIZE = (512, 512)

    def __init__(self, data_root='data/AerialImageDataset/train/train', mode='train', img_dir='images', mask_dir='masks',
                 img_suffix='.png', mask_suffix='.png', mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = InriaDataset.get_validation_transform() if mode.lower() != 'train' else InriaDataset.get_training_transform()
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, inputs=img, mask=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(os.path.join(data_root, img_dir))
        mask_filename_list = os.listdir(os.path.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = os.path.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.float32)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = A.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = A.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = A.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = A.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        return img, mask


    def get_training_transform():
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize()
        ]
        return A.Compose(train_transform)


    def get_validation_transform():
        val_transform = [
            A.Normalize()
        ]
        return A.Compose(val_transform)


class MassBuildDataset(Dataset):
    
    CLASSES = ('Building', 'Background')
    PALETTE = [[255, 255, 255],  [0, 0, 0]]
    ORIGIN_IMG_SIZE = (1500, 1500)
    INPUT_IMG_SIZE = (1536, 1536)
    TEST_IMG_SIZE = (1500, 1500)

    def __init__(self, data_root='/home/oguz/cv-projects/datasets/Mass-Building', mode='train', img_dir='train_images', mask_dir='train_masks',
                 img_suffix='.png', mask_suffix='.png', mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = MassBuildDataset.get_validation_transform() if mode.lower() != 'train' else MassBuildDataset.get_training_transform()
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask / 255).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, inputs=img, mask=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(os.path.join(data_root, img_dir))
        mask_filename_list = os.listdir(os.path.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = os.path.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = A.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = A.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = A.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = A.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        return img, mask


    def get_training_transform():
        train_transform = [
            A.RandomRotate90(p=0.5),
            A.RandomCrop(height=1024, width=1024, p=1.0),
            A.Normalize()
        ]
        return A.Compose(train_transform)


    def get_validation_transform():
        val_transform = [
            A.PadIfNeeded(min_height=1536, min_width=1536, position="top_left",
                            border_mode=0, value=[0, 0, 0], mask_value=[255, 255, 255]),
            A.Normalize()
        ]
        return A.Compose(val_transform)


class WHUBuildingDataset(Dataset):
    CLASSES = ('Building', 'Background')
    PALETTE = [[255, 255, 255],  [0, 0, 0]]

    ORIGIN_IMG_SIZE = (512, 512)
    INPUT_IMG_SIZE = (512, 512)
    TEST_IMG_SIZE = (512, 512)

    def __init__(self, data_root='/home/oguz/cv-projects/datasets/WHU', mode='train', img_dir='Image', mask_dir='Mask',
                 img_suffix='.tif', mask_suffix='.png', mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = WHUBuildingDataset.get_validation_transform() if mode.lower() != 'train' else WHUBuildingDataset.get_training_transform()
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask / 255).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, inputs=img, mask=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(os.path.join(data_root, img_dir))
        mask_filename_list = os.listdir(os.path.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = os.path.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return np.array(img), np.array(mask)

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = A.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = A.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = A.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = A.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)

        return img, mask

    def get_training_transform():
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize()
        ]
        return A.Compose(train_transform)


    def train_aug(img, mask):
        # crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        #                     SmartCropV1(crop_size=384, max_ratio=0.5, ignore_index=len(CLASSES), nopad=False)])
        # img, mask = crop_aug(img, mask)
        img, mask = np.array(img), np.array(mask)
        aug = WHUBuildingDataset.get_training_transform()(image=img.copy(), mask=mask.copy())
        img, mask = aug['image'], aug['mask']
        return img, mask


    def get_validation_transform():
        val_transform = [
            A.Normalize()
        ]
        return A.Compose(val_transform)
    
    def val_aug(img, mask):
        img, mask = np.array(img), np.array(mask)
        aug = WHUBuildingDataset.get_val_transform()(image=img.copy(), mask=mask.copy())
        img, mask = aug['image'], aug['mask']
        return img, mask