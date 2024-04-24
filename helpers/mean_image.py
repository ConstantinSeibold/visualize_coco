import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def plot_mean_image(img, image_root, out_path_indv):
    filename = f"{out_path_indv}/mean_image/mean_image.png"
    os.makedirs(f"{out_path_indv}/mean_image/", exist_ok=True)

    mean_img = np.zeros((512,512,3))
    for id in tqdm(range(len(img))):
        # import pdb; pdb.set_trace()
        image = img.iloc[id]
        # print(image)
        i = Image.open(os.path.join(image_root, image["file_name"])).convert("RGB").resize((512,512))
        mean_img += np.array(i)
    mean_img = mean_img / len(img)
    mean_img = mean_img.astype(np.uint8)
    Image.fromarray(mean_img).save(filename)

    out_json = [
        {
            "category_name": "Mean Image",
            "supercategory_name": "",
            "filePaths": filename
        }
    ]

    
    with open(f"{out_path_indv}/mean_image.json", "w") as outfile:
        json.dump(out_json, outfile)
        
    return [filename]