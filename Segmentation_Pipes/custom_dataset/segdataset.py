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
      



class FewShotSegDataset(Dataset):
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
        #========================================================================
        import collections
        from itertools import product
        
        sub_path = "train" if self.train else "valid"
        
        json_file = os.path.join(root_dir, sub_path, '_annotations.coco.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        Img_Dict = collections.defaultdict(list)
        for anno in data['annotations']:
            #print(anno)
            Img_Dict[anno['image_id']].append(anno['category_id'])
            
        single_label_imgs = []
        multi_label_imgs = []
        for k, v in Img_Dict.items():
            #print(k, v)
            if len(set(v)) == 1 :
                single_label_imgs.append(k)
            else:
                multi_label_imgs.append(k)
            #Img_Dict[k] = len(set(v))
        #combinations = list(product(single_label_imgs, multi_label_imgs))
        
        single_label_imgs_names = []
        multi_label_imgs_names = []
        for img_ in data['images']:
            if img_['id'] in single_label_imgs:
                #print(img_)
                single_label_imgs_names.append(img_['file_name'].split(".")[0].replace("_png",".png"))
            elif img_['id'] in multi_label_imgs:
                multi_label_imgs_names.append(img_['file_name'].split(".")[0].replace("_png",".png"))
        self.combinations_names = list(product(single_label_imgs_names, multi_label_imgs_names))

        #=========

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    
    
    def convert_qry_label(self,sup_sample, qry_sample):
        '''
        support label 은 1가지 레이블만 존재하는 이미지들로 골랐고,
        query label 은 다양한 레이블이 존재하지만 support label 에 있는 레이블과 동일한 레이블만 1로 만들었다.
        
        추후에 multiclass 를 query 한장으로 예측하기 위해서는 support label 을 그대로 두면 되지만,
        지금은 support label 에 대한 binary 예측을 위해서 support label 도 다시 binary mask 로 만들었다. 
        '''
        
        sup_unique = [i for i in torch.unique(sup_sample) if i != 0][0]

        #-- mask 
        mask = (qry_sample == sup_unique)
        reverse_mask = (mask == False)

        qry_sample[mask] = 1
        qry_sample[reverse_mask] = 0
        
        # support label to binary 
        sup_mask = (sup_sample != 0)
        sup_sample[sup_mask] = 1
        
        return sup_sample, qry_sample 

    def __getitem__(self, idx):
                
        support_name = self.combinations_names[idx][0]
        query_name   = self.combinations_names[idx][1]

        #--- support
        support_image = Image.open(os.path.join(self.img_dir, support_name))
        support_segmentation_map = Image.open(os.path.join(self.ann_dir, support_name))

        # randomly crop + pad both image and segmentation map to same size
        support_encoded_inputs = self.image_processor(support_image, support_segmentation_map, return_tensors="pt")

        for k,v in support_encoded_inputs.items():
            support_encoded_inputs[k].squeeze_() # remove batch dimension
          
        #--- query
        query_image = Image.open(os.path.join(self.img_dir, query_name))
        query_segmentation_map = Image.open(os.path.join(self.ann_dir, query_name))

        # randomly crop + pad both image and segmentation map to same size
        query_encoded_inputs = self.image_processor(query_image, query_segmentation_map, return_tensors="pt")

        for k,v in query_encoded_inputs.items():
            query_encoded_inputs[k].squeeze_() # remove batch dimension          
        
        
        #--- support label 의 class 와 동일한 query label 의 class 만 1 로 만들고 나머지는 0 마스킹
        support_labels = support_encoded_inputs['labels']
        query_labels = query_encoded_inputs['labels']
        s_l, q_l_binary = self.convert_qry_label(support_labels,query_labels)

        support_encoded_inputs['labels'] = s_l
        query_encoded_inputs['labels'] = q_l_binary

        return support_encoded_inputs, query_encoded_inputs
      