#https://youtu.be/SQng3eIEw-k
"""

Convert coco json labels to labeled masks and copy original images to place 
them along with the masks. 

Dataset from: https://github.com/sartorius-research/LIVECell/tree/main
Note that the dataset comes with: 
Creative Commons Attribution - NonCommercial 4.0 International Public License
In summary, you are good to use it for research purposes but for commercial
use you need to investigate whether trained models using this data must also comply
with this license - it probably does apply to any derivative work so please be mindful. 

You can directly download from the source github page. Links below.

Training json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
Validation json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json
Test json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
Images: Download images.zip by following the link: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

If these links do not work, follow the instructions on their github page. 


"""

import json
import numpy as np
import skimage
import tifffile
import os
import shutil


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    # Counter for the object number
    #object_number = 1

    #--- category table => 백그라운드를 추가하므로, 백그라운드가 0이됨. 그뒤에 레이블 +1 
    #     "categories": [
    #     {
    #         "id": 0,
    #         "name": "M2A1Slammer",
    #         "supercategory": "M2A1Slammer"
    #     },
    #     {
    #         "id": 1,
    #         "name": "M5SandstormMLRS",
    #         "supercategory": "M2A1Slammer"
    #     },
    #     {
    #         "id": 2,
    #         "name": "T140Angara",
    #         "supercategory": "M2A1Slammer"
    #     },
    #     {
    #         "id": 3,
    #         "name": "ZamakMRL",
    #         "supercategory": "M2A1Slammer"
    #     }
    # ],
    #---    


    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            ann['category_id'] +=1
            category_id = ann['category_id'] 
            
            for idx,seg in enumerate(ann['segmentation']):
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = category_id
                
                #object_number += 1 #We are assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a TIFF using tifffile library
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.tif', '_mask.tif').replace(".png","_mask.png"))
    tifffile.imsave(mask_path, mask_np)
    

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")

    return annotations


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    print(categories)
    #--------
    # category handling
    for ca in categories:
        print(ca)
        ca["id"] = int(ca["id"]) + 1
    
    #--------
    new_category = {"id": 0, "name": "background", "supercategory": "background"}
    # Insert the new category at the beginning of the categories list
    categories.insert(0, new_category)
    #--------
    

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        # Create the masks
        create_mask(img, annotations, mask_output_folder)
        
        
        # Copy original images to the specified folder
        #original_image_path = os.path.join(original_image_dir, img['file_name'])
    
        # new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        # shutil.copy2(original_image_path, new_image_path)
        # print(f"Copied original image to {new_image_path}")
        
    #--
    #print(data)
   
    json_string = json.dumps(data, default=str)
    # with open( "/home/eric/srcs/FewShotSeg_Lab/FewShotVision_Lab/fixed_anno.json", 'w') as file:
    #     json.dump(data, file)
    with open("/home/eric/srcs/FewShotSeg_Lab/FewShotVision_Lab/fixed_anno.json", "w") as file:
        file.write(json_string)  #


if __name__ == '__main__':
    original_image_dir = '/disk3/eric/dataset/VISION_SOFS/WEAPON_4/segmentation_pipe/images'  # Where your original images are stored
    json_file = '/disk3/eric/dataset/VISION_SOFS/WEAPON_4/segmentation_pipe/_annotations.coco.json'
    
    mask_output_folder = '/disk3/eric/dataset/VISION_SOFS/WEAPON_4/segmentation_pipe/labels'  # Modify this as needed. Using val2 so my data is not overwritten
    image_output_folder = '/disk3/eric/dataset/VISION_SOFS/WEAPON_4/train_mask'  # 
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)


