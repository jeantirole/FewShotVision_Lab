import json
from huggingface_hub import hf_hub_download
import os 
from PIL import Image
import numpy as np


from torch.utils.data import Dataset
import os
from PIL import Image

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class InstanceSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "train" if self.train else "valid"
        
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.ann_dir = os.path.join(self.root_dir, sub_path, "labels")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        #print(len(self.images))
        #print(len(self.annotations))

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
      
  

class PairwiseImageRetrievalDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            augmentation (callable, optional): Optional augmentation to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.aug = augmentation
        
        # Step 1: Load all images and assign class labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Go through each class folder
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)
        
        # Create a dictionary of image paths categorized by class
        self.class_images = {}
        for img_path, label in zip(self.image_paths, self.labels):
            if label not in self.class_images:
                self.class_images[label] = []
            self.class_images[label].append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            A pair of images, one positive and one negative.
            - image1: A sample image
            - image2: A positive or negative pair
            - label: 1 if positive pair, 0 if negative pair
        """
        # Randomly select the first image (image1)
        img_path1 = self.image_paths[idx]
        label1 = self.labels[idx]
        
        # Load the first image
        image1 = Image.open(img_path1).convert('RGB')
        image1 = np.asarray(image1)
        
        # Apply the augmentations and transformations for image1
        if self.aug:
            image1 = self.aug(image=image1)['image']  # Albumentations requires dict output
        if self.transform:
            image1 = self.transform(image1)
        
        # Randomly select a pair: Positive or Negative
        # Positive pair: same class
        # Negative pair: different class
        is_positive_pair = random.choice([True, False])
        
        if is_positive_pair:
            # Select a positive pair: image from the same class
            positive_img_path = random.choice(self.class_images[label1])
            while positive_img_path == img_path1:  # Ensure it's not the same image
                positive_img_path = random.choice(self.class_images[label1])
            
            image2 = Image.open(positive_img_path).convert('RGB')
            label = 1  # Positive pair
        else:
            # Select a negative pair: image from a different class
            negative_label = random.choice([label for label in self.class_images if label != label1])
            negative_img_path = random.choice(self.class_images[negative_label])
            
            image2 = Image.open(negative_img_path).convert('RGB')
            label = 0  # Negative pair
        image2 = np.asarray(image2)

        # Apply the augmentations and transformations for image2
        if self.aug:
            image2 = self.aug(image=image2)['image']
        if self.transform:
            image2 = self.transform(image2)

        return image1, image2, label