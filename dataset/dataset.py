from __future__ import annotations
import numpy as np
import os
from PIL import Image
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchxrayvision as xrv


class SmallDataset:
    """
    A dataset class for loading and preprocessing small size imaging datasets.
    It supports loading images and masks from directories, resizing, padding, 
    and basic or specialized normalization (ImageNet, TorchXRayVision).
    """
    def __init__(self, images_path: str, masks_path: str =None, size: int =256, normalization: str ="basic"):
        """
        Initializes the SmallDataset with images and optional masks.

        Args:
            images_path (str): Path to the directory containing image files.
            masks_path (str, optional): Path to the directory containing mask files. Defaults to None.
            size (int, optional): The target size (width and height) for resizing and padding. Defaults to 256.
            normalization (str, optional): The type of normalization to apply. Defaults to "basic".
        
        Raises:
            ValueError: If the number of images and masks do not match.
        """
        self.normalization = normalization

        self.images = SmallDataset.load_images_to_list(images_path, size=size)
        if masks_path:
            self.masks = SmallDataset.load_images_to_list(masks_path, size=size)
        else:
            # Create empty masks
            self.masks = [np.zeros((size, size), dtype=np.uint8) for _ in range(len(self.images))]

        if len(self.images) != len(self.masks):
            raise ValueError("Images and masks must have the same length.")
        
        self.images = np.stack(self.images)
        self.masks = np.stack(self.masks)
        self.basic_normalization()
        
        self.folds = []


    # ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)
    MEAN = 0.485
    STD = 0.229


    def __len__(self) -> int:
        return self.images.shape[0]


    @staticmethod
    def load_images_to_list(path: str, size: int =256) -> list[np.ndarray]:
        """
        Loads all PNG images from a directory, resizes and pads them to a target size.

        Args:
            path (str): Path to the directory containing image files.
            size (int, optional): The target size for resizing and padding. Defaults to 256.

        Returns:
            list[np.ndarray]: A list of processed grayscale images as numpy arrays.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        out = []

        for filename in sorted(os.listdir(path)):
            if filename.endswith(".png"):
                file_path = os.path.join(path, filename)
                # Load grayscale image
                image = np.array(Image.open(file_path).convert("L"))

                image = SmallDataset.resize_and_pad(image, size=size)

                out.append(np.array(image))
        return out
    

    @staticmethod
    def resize_and_pad(img: np.ndarray, size: int =256) -> np.ndarray:
        """
        Resizes the given image to the given size, and pads it to fit the size if necessary.

        Args:
            img (np.ndarray): The image to be resized and padded.
            size (int, optional): The size to resize the image to. Defaults to 256.

        Returns:
            np.ndarray: The resized and padded image.
        """
        height, width = img.shape
        scale = size / max(height, width)
        new_w, new_h = int(width * scale), int(height * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image with black background
        new_img = np.full((size, size), 0, dtype=img.dtype)
        
        # Center the resized image
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        new_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return new_img


    @staticmethod
    def special_normalize_image(img: torch.Tensor, normalization: str) -> torch.Tensor:
        """
        Normalize the given image based on the given normalization type.

        Args:
            img (torch.Tensor): The image to be normalized.
            normalization (str): The normalization type to use.
                - "imagenet": Normalize using the mean and standard deviation of the ImageNet dataset.
                - "xrv": Normalize using the normalization of TorchXRayVision.

        Returns:
            torch.Tensor: The normalized image.
        """
        img = img.numpy()

        if normalization == "imagenet":
            img = img / 255
            img = (img - SmallDataset.MEAN) / SmallDataset.STD

            return torch.from_numpy(img)
        # TorchXRayVision normalization
        elif normalization == "xrv":
            img = xrv.datasets.normalize(img, 1)
            
            return torch.from_numpy(img)
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")


    def cross_validation_split(self, folds: int =1, shuffle: bool =True):
        """
        Split the dataset into folds for cross-validation.

        Args:
            folds (int, optional): The number of folds to split the dataset into. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.
        """
        if folds == 1:
            fold_images = torch.from_numpy(self.images).float()
            fold_masks = torch.from_numpy(self.masks).float()
            self.folds.append(Fold(fold_images, fold_masks, self.normalization))
            return

        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        splits = np.array_split(indices, folds)

        for fold_indices in splits:
            fold_images = torch.from_numpy(self.images[fold_indices]).float()
            fold_masks = torch.from_numpy(self.masks[fold_indices]).float()
            self.folds.append(Fold(fold_images, fold_masks, self.normalization))


    def basic_normalization(self):
        """
        Basic [0, 1] normalization.
        """
        self.images = self.images / 255
        self.masks = self.masks / 255

            
class Fold(Dataset):
    """
    A Dataset class representing a specific fold of data for PyTorch.
    This class handles image and mask storage, normalization, and data augmentation.
    """
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, normalization: str ="basic", augment: bool =False):
        """
        Fold class prepared for pytorch dataloader.

        Args:
            images (torch.Tensor): The images of the fold.
            masks (torch.Tensor): The masks of the fold.
            normalization (str, optional): Type of normalization to apply to the images and masks. Defaults to "basic" [0, 1].
            augment (bool, optional): Whether to apply data augmentation to the sample. Defaults to False.
        """
        # Add a channel dimension
        self.images = images.unsqueeze(1).float()
        self.masks = masks.unsqueeze(1).long()

        self.normalization = normalization
        self.augment = augment


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Contains logic for augmentation and special normalization.
        """
        if self.augment:
            # Apply data augmentation
            aug_img, aug_mask = self.transform(self.images[index], self.masks[index])
            
            if self.normalization != "basic":
                return SmallDataset.special_normalize_image(aug_img, self.normalization), aug_mask
            else:
                # Basic [0, 1] normalization
                return aug_img, aug_mask
        else:
            if self.normalization != "basic":
                return SmallDataset.special_normalize_image(self.images[index], self.normalization), self.masks[index]
            else:
                # Basic [0, 1] normalization
                return self.images[index], self.masks[index]
    

    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies random transformations to the given image and mask.

        Args:
            image (torch.Tensor): The image to be transformed.
            mask (torch.Tensor): The mask to be transformed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image and mask.
        """
        # Geometry
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        # Pixel level
        if random.random() < 0.5:
            # Brightness
            factor = random.uniform(0.8, 1.2)
            image = image * factor

        # Gaussian noise
        if random.random() < 0.5:
            image = image + torch.randn_like(image) * 0.02  # std = 0.02

        # Keep image normalized
        image = image.clamp(0, 1)

        return image, mask
    

    @staticmethod
    def combine_folds(folds: list[Fold]) -> Fold:
        """
        Combine multiple folds into one.

        Args:
            folds (list[Fold]): A list of folds to combine.

        Returns:
            Fold: A new fold containing all the images and masks from the input folds.

        Raises:
            ValueError: If the normalization of the input folds is not the same.
        """
        # At least 1 augmentation = augmentation for result
        augment = any(fold.augment for fold in folds)
        # Normalization
        if len({fold.normalization for fold in folds}) > 1:
            raise ValueError("All folds must have the same normalization")
        normalization = list(folds)[0].normalization
        
        images = torch.cat([fold.images for fold in folds], dim=0).squeeze(1)
        masks = torch.cat([fold.masks for fold in folds], dim=0).squeeze(1)

        return Fold(images, masks, normalization, augment)
    

    @staticmethod
    def split_fold(fold: Fold, ratio: float=0.5) -> tuple[Fold, Fold]:
        """
        Split a fold into two folds based on the given ratio.

        Args:
            fold (Fold): The fold to be split.
            ratio (float, optional): The ratio of the first fold to the total size. Defaults to 0.5.

        Returns:
            tuple[Fold, Fold]: A tuple containing the two folds resulting from the split.
        """
        augment = fold.augment
        normalization = fold.normalization
        images = fold.images.squeeze(1)
        masks = fold.masks.squeeze(1)

        size = int(len(images) * ratio)

        return Fold(images[:size], masks[:size], normalization, augment), Fold(images[size:], masks[size:], normalization, augment)
