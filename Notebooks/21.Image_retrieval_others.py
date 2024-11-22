import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


images = os.listdir("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop")

os.environ["TORCH_HOME"] = "/disk3/eric/weight"
model = torchvision.models.resnet18(pretrained=True)

all_names = []
all_vecs = None
model.eval()
root = "/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/"

transform = transforms.Compose([
    transforms.Resize((256 , 256)) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
])

activation = {}
def get_activation(name):
    def hook(model , input , output):
        activation[name] = output.detach()
    return hook



model.avgpool.register_forward_hook(get_activation("avgpool"))

with torch.no_grad():
    for i , file in enumerate(images):
            img = Image.open(root + file)
            img = transform(img)
            out = model(img[None , ...])
            vec = activation["avgpool"].numpy().squeeze()[None , ...]
            if all_vecs is None:
                all_vecs = vec
            else:
                all_vecs = np.vstack([all_vecs , vec])
            all_names.append(file)
    if i % 100 == 0 and i != 0:
        print(i , "done")

#------------------------------
# result save

np.save("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_vecs.npy" , all_vecs)
np.save("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_names.npy" , all_names)


import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist




@st.cache_data
def read_data():
    all_vecs = np.load("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_vecs.npy",allow_pickle=True)
    all_names = np.load("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_names.npy",allow_pickle=True)
    return all_vecs , all_names

vecs , names = read_data()

_ , fcol2 , _ = st.columns(3)

scol1 , scol2 = st.columns(2)

ch = scol1.button("Start / change")
fs = scol2.button("find similar")

if ch:
    random_name = names[np.random.randint(len(names))]
    fcol2.image(Image.open( root +  random_name))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])
if fs:
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]
    fcol2.image(Image.open("./images/" + st.session_state["disp_img"]))
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
    c1.image(Image.open(root + names[top5[0]]))
    c2.image(Image.open(root + names[top5[1]]))
    c3.image(Image.open(root + names[top5[2]]))
    c4.image(Image.open(root + names[top5[3]]))
    c5.image(Image.open(root + names[top5[4]]))