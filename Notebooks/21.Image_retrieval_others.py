{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/eric/anaconda3/envs/trex/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision\n",
    "from torchvision import transforms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "images = os.listdir(\"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop\")\n",
    "\n",
    "os.environ[\"TORCH_HOME\"] = \"/disk3/eric/weight\"\n",
    "model = torchvision.models.resnet18(weights = \"DEFAULT\")\n",
    "\n",
    "all_names = []\n",
    "all_vecs = None\n",
    "model.eval()\n",
    "root = \"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/\"\n",
    "\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((256 , 256)) ,\n",
    "    transforms.ToTensor() ,\n",
    "    transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])\n",
    "])\n",
    "\n",
    "activation = {}\n",
    "def get_activation(name):\n",
    "    def hook(model , input , output):\n",
    "        activation[name] = output.detach()\n",
    "    return hook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<torch.utils.hooks.RemovableHandle at 0x7fa1309f52b0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.avgpool.register_forward_hook(get_activation(\"avgpool\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_7.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_8.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_1.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_6.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_3.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_0.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_9.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_5.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_4.png\n",
      "filename /disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_crop/cropped_box_2.png\n"
     ]
    }
   ],
   "source": [
    "model.avgpool.register_forward_hook(get_activation(\"avgpool\"))\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i , file in enumerate(images):\n",
    "            img = Image.open(root + file)\n",
    "            img = transform(img)\n",
    "            out = model(img[None , ...])\n",
    "            vec = activation[\"avgpool\"].numpy().squeeze()[None , ...]\n",
    "            if all_vecs is None:\n",
    "                all_vecs = vec\n",
    "            else:\n",
    "                all_vecs = np.vstack([all_vecs , vec])\n",
    "            all_names.append(file)\n",
    "    if i % 100 == 0 and i != 0:\n",
    "        print(i , \"done\")\n",
    "\n",
    "#------------------------------\n",
    "# result save\n",
    "\n",
    "np.save(\"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_vecs.npy\" , all_vecs)\n",
    "np.save(\"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_names.npy\" , all_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-21 10:51:10.290 No runtime found, using MemoryCacheStorageManager\n",
      "2024-11-21 10:51:10.292 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.292 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.292 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.293 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.294 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.294 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.294 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.295 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.295 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.295 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.296 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.296 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.296 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.296 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.298 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.298 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.299 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-21 10:51:10.299 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "import time\n",
    "from scipy.spatial.distance import cdist\n",
    "\n",
    "@st.cache_data\n",
    "def read_data():\n",
    "    all_vecs = np.load(\"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_vecs.npy\",allow_pickle=True)\n",
    "    all_names = np.load(\"/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result_z_cos/all_names.npy\",allow_pickle=True)\n",
    "    return all_vecs , all_names\n",
    "\n",
    "vecs , names = read_data()\n",
    "\n",
    "_ , fcol2 , _ = st.columns(3)\n",
    "\n",
    "scol1 , scol2 = st.columns(2)\n",
    "\n",
    "ch = scol1.button(\"Start / change\")\n",
    "fs = scol2.button(\"find similar\")\n",
    "\n",
    "if ch:\n",
    "    random_name = names[np.random.randint(len(names))]\n",
    "    fcol2.image(Image.open( root +  random_name))\n",
    "    st.session_state[\"disp_img\"] = random_name\n",
    "    st.write(st.session_state[\"disp_img\"])\n",
    "if fs:\n",
    "    c1 , c2 , c3 , c4 , c5 = st.columns(5)\n",
    "    idx = int(np.argwhere(names == st.session_state[\"disp_img\"]))\n",
    "    target_vec = vecs[idx]\n",
    "    fcol2.image(Image.open(\"./images/\" + st.session_state[\"disp_img\"]))\n",
    "    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]\n",
    "    c1.image(Image.open(root + names[top5[0]]))\n",
    "    c2.image(Image.open(root + names[top5[1]]))\n",
    "    c3.image(Image.open(root + names[top5[2]]))\n",
    "    c4.image(Image.open(root + names[top5[3]]))\n",
    "    c5.image(Image.open(root + names[top5[4]]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "trex",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
