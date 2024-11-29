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

#========================================================================
import collections
from itertools import product


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
        print(self.combinations_names)
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
    
    def my_collate_fn(self,batch):
        # Support와 Query를 각각 묶음
        support_batch = {k: torch.stack([b["support"][k] for b in batch]) for k in batch[0]["support"]}
        query_batch = {k: torch.stack([b["query"][k] for b in batch]) for k in batch[0]["query"]}

        return {
            "support": support_batch,
            "query": query_batch,
        }

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

        #-- 
        #sup_n_query = [support_encoded_inputs, query_encoded_inputs]
        return {
            "support": {
                "images": support_encoded_inputs['pixel_values'],
                "labels": support_encoded_inputs['labels'],
            },
            "query": {
                "images": query_encoded_inputs['pixel_values'],
                "labels": query_encoded_inputs['labels'],
            }
        }        




class FewShotSegDataset_v1(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, n_shots, n_instance,train=True):
        """
        1127 
        few-shot 의 metric 을 실험하기 위해서, 1-shot, 5-shot, 10-shot 으로 나눠서 진행. 
        1-shot 과, 5-shot 은 딱맞게 장수가 있으나 10-shot 은 약간씩 부족. 
        n_instance 변수는, 각 support 이미지에 있는 instance(object) 의 수를 의미함. 
        지금은 각 이미지당 1개의 객체가 있을 때를 가정하였음. 


        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "train" if self.train else "valid"
            
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.ann_dir = os.path.join(self.root_dir, sub_path, "labels")
        
        
        json_file = os.path.join(root_dir, sub_path, '_annotations.coco.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        Img_Dict = collections.defaultdict(list)
        for anno in data['annotations']:
            #print(anno)
            Img_Dict[anno['image_id']].append(anno['category_id'])
            
        single_label_imgs = []
        multi_label_imgs = []

        class_cnt_dict = {
        "1" : 0, 
        "2" : 0,
        "3" : 0,
        "4" : 0}

        n_instance = n_instance
        n_shots_ = n_shots
        for k, v in Img_Dict.items():
            #print(k, v)
            # class 의 종류가 하나
            if (len(set(v)) == n_instance) and (len(v)==n_instance) :        
                if class_cnt_dict[str(v[0])] < n_shots_:
                    single_label_imgs.append(k)
                    class_cnt_dict[str(v[0])] +=1  
            else:
                multi_label_imgs.append(k)
        
        self.single_label_imgs_names = []
        self.multi_label_imgs_names = []
        for img_ in data['images']:
            if img_['id'] in single_label_imgs:
                #print(img_)
                self.single_label_imgs_names.append(img_['file_name'].split(".")[0].replace("_png",".png"))
            elif img_['id'] in multi_label_imgs:
                self.multi_label_imgs_names.append(img_['file_name'].split(".")[0].replace("_png",".png"))
        self.combinations_names = list(product(self.single_label_imgs_names, self.multi_label_imgs_names))
        print(self.combinations_names)
        #=========

        
    def __len__(self):
        return len(self.combinations_names)
    
    
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
    
    def my_collate_fn(self,batch):
        # Support와 Query를 각각 묶음
        support_batch = {k: torch.stack([b["support"][k] for b in batch]) for k in batch[0]["support"]}
        query_batch = {k: torch.stack([b["query"][k] for b in batch]) for k in batch[0]["query"]}

        return {
            "support": support_batch,
            "query": query_batch,
        }

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

        #-- 
        #sup_n_query = [support_encoded_inputs, query_encoded_inputs]
        return {
            "support": {
                "images": support_encoded_inputs['pixel_values'],
                "labels": support_encoded_inputs['labels'],
            },
            "query": {
                "images": query_encoded_inputs['pixel_values'],
                "labels": query_encoded_inputs['labels'],
            }
        }      
        
        


class FewShotSegDataset_v2(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, n_shots, n_instance,train=True):
        """
        1127 
        few-shot 의 metric 을 실험하기 위해서, 1-shot, 5-shot, 10-shot 으로 나눠서 진행. 
        1-shot 과, 5-shot 은 딱맞게 장수가 있으나 10-shot 은 약간씩 부족. 
        n_instance 변수는, 각 support 이미지에 있는 instance(object) 의 수를 의미함. 
        지금은 각 이미지당 1개의 객체가 있을 때를 가정하였음. 
        
        
        1128
        class 별로 학습할 수 있도록 데이터 뽑는 구조 변경 필요함.
        validation 시에 좀더 다양한 데이터셋을 볼 수 있도록 변경 필요함. 
        train 과 validation 의 개념을 바꿨기 때문에. img 와 ann 을 불러올 때, sub_path 가 바뀌었음.


        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        self.sub_path = "train" if self.train else "valid"
            
        self.img_dir = os.path.join(self.root_dir, "train", "images")
        self.ann_dir = os.path.join(self.root_dir, "train", "labels")
        
        
        json_file = os.path.join(root_dir, "train", '_annotations.coco.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        Img_Dict = collections.defaultdict(list)
        for anno in data['annotations']:
            #print(anno)
            Img_Dict[anno['image_id']].append(anno['category_id'])
            
        single_label_imgs = []
        multi_label_imgs = []

        class_cnt_dict = {
        "1" : 0, 
        "2" : 0,
        "3" : 0,
        "4" : 0}

        n_instance = n_instance
        n_shots_ = n_shots
        for k, v in Img_Dict.items():
            #print(k, v)
            # class 의 종류가 하나
            if (len(set(v)) == n_instance) and (len(v)==n_instance) :        
                if class_cnt_dict[str(v[0])] < n_shots_:
                    single_label_imgs.append(k)
                    class_cnt_dict[str(v[0])] +=1  
            else:
                multi_label_imgs.append(k)
        
        self.single_label_imgs_names = []
        self.multi_label_imgs_names = []


        for id_ in single_label_imgs:
            self.single_label_imgs_names.append(data['images'][id_]['file_name'].split(".")[0].replace("_png",".png"))

        for id_ in multi_label_imgs:
            self.multi_label_imgs_names.append(data['images'][id_]['file_name'].split(".")[0].replace("_png",".png"))

        #-- 여기서 single label 
        combinations_names = list(product(self.single_label_imgs_names, self.multi_label_imgs_names))
        #print(combinations_names)
        #print(self.combinations_names)
        #=========
        # few-shot class 별로 구분
        self.combinations_names_for_fewshot_train = []

        for name in self.single_label_imgs_names:
            for name2 in self.single_label_imgs_names:
                if (name.split("__")[0] == name2.split("__")[0]) and (name.split("__") != name2.split("__")):
                    self.combinations_names_for_fewshot_train.append([name,name2])

        self.combinations_names_for_fewshot_valid = []

        for name in sorted(self.single_label_imgs_names):
            for name_ in self.multi_label_imgs_names:
                self.combinations_names_for_fewshot_valid.append([name,name_])

        print("train pairs : ", self.combinations_names_for_fewshot_train)
        print("valid pairs ; ", self.combinations_names_for_fewshot_valid)
    
        #=========
        
    def __len__(self):
        if self.sub_path == "train":
            return (len(self.combinations_names_for_fewshot_train))
        if self.sub_path == "valid":
            return (len(self.combinations_names_for_fewshot_valid))
    
    
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
    
    def my_collate_fn(self,batch):
        # Support와 Query를 각각 묶음
        support_batch = {k: torch.stack([b["support"][k] for b in batch]) for k in batch[0]["support"]}
        query_batch = {k: torch.stack([b["query"][k] for b in batch]) for k in batch[0]["query"]}

        return {
            "support": support_batch,
            "query": query_batch,
        }

    def __getitem__(self, idx):
        
        if self.sub_path == "train":
            support_name = self.combinations_names_for_fewshot_train[idx][0]
            query_name   = self.combinations_names_for_fewshot_train[idx][1]
        elif self.sub_path == "valid":
            support_name = self.combinations_names_for_fewshot_valid[idx][0]
            query_name   = self.combinations_names_for_fewshot_valid[idx][1]            


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

        #-- 
        #sup_n_query = [support_encoded_inputs, query_encoded_inputs]
        return {
            "support": {
                "images": support_encoded_inputs['pixel_values'],
                "labels": support_encoded_inputs['labels'],
            },
            "query": {
                "images": query_encoded_inputs['pixel_values'],
                "labels": query_encoded_inputs['labels'],
            }
        }        