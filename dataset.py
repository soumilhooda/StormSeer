import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import logging

class CycloneDataset(Dataset):
    def __init__(self, image_paths, speed_labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.speed_labels = speed_labels
        self.transform = transform
        self.augment = augment
        self.logger = logging.getLogger(self.__class__.__name__)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        speed = self.speed_labels[idx]

        if self.augment:
            image = self.apply_augmentations(image)

        if self.transform:
            image = self.transform(image)

        self.logger.info(f"Loaded image from {img_path}, shape: {image.shape}, speed: {speed}")
        return image, speed

    def apply_augmentations(self, image):
        self.logger.info("Applying augmentations")
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = transforms.functional.rotate(image, angle)
            self.logger.info(f"Applied rotation with angle {angle}")

        # Random horizontal flip
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            self.logger.info("Applied horizontal flip")

        # Random vertical flip
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            self.logger.info("Applied vertical flip")

        # Random brightness and contrast
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)
            image = transforms.functional.adjust_contrast(image, contrast_factor)
            self.logger.info(f"Adjusted brightness ({brightness_factor}) and contrast ({contrast_factor})")

        # Random Gaussian noise
        if random.random() > 0.5:
            image = np.array(image)
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
            self.logger.info("Added Gaussian noise")

        # Random color jitter
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = color_jitter(image)
            self.logger.info("Applied color jitter")

        # Random perspective transform
        if random.random() > 0.5:
            perspective_transform = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
            image = perspective_transform(image)
            self.logger.info("Applied perspective transform")

        return image