import argparse
import os
import json 
import numpy as np
from PIL import Image

from trex import TRex2APIWrapper, visualize


def get_args():
    parser = argparse.ArgumentParser(description="Interactive Inference")
    parser.add_argument(
        "--token",
        type=str,
        help="The token for T-Rex2 API. We are now opening free API access to T-Rex2",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="The threshold for box score"
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="demo_vis/",
        help="The directory for visualization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trex2 = TRex2APIWrapper(args.token)
    
    prompts = [
        {
            "prompt_image": "/disk3/eric/dataset/VISION_SOFS/WEAPON_4/test_splits/north2_part_8.png",
            "rects":    [[485, 220, 546, 265]]
        },
        # {
        #     "prompt_image": "assets/trex2_api_examples/generic_prompt2.jpg",
        #     "rects": [[561, 231, 634, 351]],
        # },
    ]
    
    #--
    
    target_image = "/disk3/eric/dataset/VISION_SOFS/WEAPON_4/test_splits/north2_part_9.png"
    
    result = trex2.generic_inference(target_image, prompts)
    # filter out the boxes with low score
    scores = np.array(result["scores"])
    labels = np.array(result["labels"])
    boxes = np.array(result["boxes"])
    filter_mask = scores > args.box_threshold
    filtered_result = {
        "scores": scores[filter_mask],
        "labels": labels[filter_mask],
        "boxes": boxes[filter_mask],
    }
    # save filter results       
    np.savetxt("/disk3/eric/dataset/VISION_SOFS/WEAPON_4/trex_result/result.txt", boxes[filter_mask], delimiter=",")

    # visualize the results
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    image = Image.open(target_image)
    image = visualize(image, filtered_result, draw_score=True)
    image.save(os.path.join(args.vis_dir, f"generic.png"))
    print(f"Visualized image saved to {args.vis_dir}/generic.png")
