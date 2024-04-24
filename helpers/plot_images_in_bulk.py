import torch, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import pandas as pd
from PIL import Image
import cv2
from pycocotools import mask as coco_mask

def plot_images_in_dr(annotations, image_df, category_df, image_root, output_path):

    embeddings = np.concatenate([
        np.stack(annotations["bbox"]), 
        np.expand_dims(np.array(annotations["area"]), 1 ) 
    ], 1)


    new_categories = []
    counter = 0
    mapping = {}
    mapping_name = {}
    remaining_cat =  category_df[category_df['id'].isin(annotations['category_id'].unique())]

    for i in range(len(remaining_cat)):
        cat = remaining_cat.iloc[i]
        mapping[cat['id']] = counter 
        mapping_name[cat['id']] = cat["name"]
        cat['id'] = counter 
        new_categories +=[cat.to_dict()]
        counter += 1

    # annotations = coco_all.loadAnns(coco_all.getAnnIds())

    # annotations = pd.DataFrame(annotations)
    category_ids = annotations['category_id']
    mapping2 = {m:mapping[m] for m in mapping}
    categories = category_ids.map(mapping2)

    categories_name = category_ids.map(mapping_name)
 
    sc = categories.unique()
    # import pdb;pdb.set_trace()


    # Assign a color to each supercategory
    file_names = []
    for category in tqdm(sc):
        name = categories_name[categories==category].unique()[0]    

        cat = category_ids[categories==category]
        data  = embeddings[categories==category]

        model = IsolationForest(contamination=0.05)
        model.fit(data)
        is_not_outlier = model.predict(data)
        annotations_id = annotations[categories==category]
        annotations_id = annotations_id[is_not_outlier == 1]

        image_ids = np.random.choice(annotations_id["image_id"].unique(), 20)
        
        image_samples = image_df[image_df["id"].isin(image_ids)]
        annotations_samples = annotations_id[annotations_id["image_id"].isin(image_ids)]

        img_list = []
        for i  in range(len(image_samples)):
            img_list += [
                            load_image_and_plot_annotations(
                                os.path.join(image_root, image_samples.iloc[i]["file_name"]), 
                                annotations_samples[annotations_samples["image_id"] == image_samples.iloc[i]["id"]])
                        ]
            
        out_img = insert_img_list_into_one(img_list)

        # import pdb; pdb.set_trace()
        file_names += [f'{output_path}/samples/{name}_samples.png']
        os.makedirs(f'{output_path}/samples/', exist_ok=True)
        Image.fromarray(out_img).save(f'{output_path}/samples/{name}_samples.png')
    
    return file_names

def plot_images_out_of_dr(annotations, image_df, category_df, image_root, output_path):

    embeddings = np.concatenate([
        np.stack(annotations["bbox"]), 
        np.expand_dims(np.array(annotations["area"]), 1 ) 
    ], 1)


    new_categories = []
    counter = 0
    mapping = {}
    mapping_name = {}
    remaining_cat =  category_df[category_df['id'].isin(annotations['category_id'].unique())]

    for i in range(len(remaining_cat)):
        cat = remaining_cat.iloc[i]
        mapping[cat['id']] = counter 
        mapping_name[cat['id']] = cat["name"]
        cat['id'] = counter 
        new_categories +=[cat.to_dict()]
        counter += 1

    # annotations = coco_all.loadAnns(coco_all.getAnnIds())

    # annotations = pd.DataFrame(annotations)
    category_ids = annotations['category_id']
    mapping2 = {m:mapping[m] for m in mapping}
    categories = category_ids.map(mapping2)

    categories_name = category_ids.map(mapping_name)
 
    sc = categories.unique()
    # import pdb;pdb.set_trace()


    # Assign a color to each supercategory
    file_names = []
    for category in tqdm(sc):
        name = categories_name[categories==category].unique()[0]    

        cat = category_ids[categories==category]
        data  = embeddings[categories==category]

        model = IsolationForest(contamination=0.05)
        model.fit(data)
        is_not_outlier = model.predict(data)

        annotations_id = annotations[categories==category]
        annotations_id = annotations_id[is_not_outlier == -1]

        image_ids = np.random.choice(annotations_id["image_id"].unique(), 20)
        
        image_samples = image_df[image_df["id"].isin(image_ids)]
        annotations_samples = annotations_id[annotations_id["image_id"].isin(image_ids)]

        img_list = []
        for i  in range(len(image_samples)):
            img_list += [
                            load_image_and_plot_annotations(
                                os.path.join(image_root, image_samples.iloc[i]["file_name"]), 
                                annotations_samples[annotations_samples["image_id"] == image_samples.iloc[i]["id"]])
                        ]
            
        out_img = insert_img_list_into_one(img_list)

        # import pdb; pdb.set_trace()
        file_names += [f'{output_path}/outlier/{name}_outlier.png']
        os.makedirs(f'{output_path}/outlier/', exist_ok=True)
        Image.fromarray(out_img).save(f'{output_path}/outlier/{name}_outlier.png')
    
    return file_names


def insert_img_list_into_one(img_list):

    out_width, out_height = 1024, 1024  # Replace with your desired dimensions

    # Initialize the output image as an array of zeros
    out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # Keep track of the starting position for each image
    current_x = 0
    current_y = 0

    for img in img_list:
        # Get the size of the current image
        img_width, img_height = img.size

        # Check if there's enough space to insert the current image
        if current_x + img_width <= out_width and current_y + img_height <= out_height:
            # Insert the image at the current position
            out_img[current_y:current_y+img_height, current_x:current_x+img_width, :] = np.array(img)

            # Update the current position for the next image
            current_x += img_width
        else:
            # Move to the next row and reset the x-coordinate
            current_x = 0
            current_y += img_height

            # Check if there's enough space in the next row
            if current_y + img_height <= out_height:
                out_img[current_y:current_y+img_height, current_x:current_x+img_width, :] = np.array(img)
                current_x += img_width
    return out_img


def load_image_and_plot_annotations(img_path, annotations):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    alpha = 0.5
    canvas = img.copy()

    # Loop through annotations
    for i in range(len(annotations)):
        annotation = annotations.iloc[i]

        if 'segmentation' in annotation and type(annotation['segmentation']) == list:
            # Polygon format
            segmentation = annotation['segmentation']
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, [np.array(segmentation, dtype=np.int32)], color=(0, 255, 0))

            cv2.polylines(canvas, [np.array(segmentation, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        elif 'segmentation' in annotation and type(annotation['segmentation']) == dict:
            # RLE format
            rle = annotation['segmentation']
            mask = coco_mask.decode(rle)
            mask = np.array(mask, dtype=np.uint8) * 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 0, 255), thickness=2)

            mask = np.stack([mask] * 3, -1)
            mask[:, :, [0, 2]] = 0
        else:
            print("Invalid segmentation format")
            continue

        img = cv2.addWeighted(canvas, 1 - alpha, img, alpha, 0)

        # Extract bounding box coordinates
        bbox = annotation['bbox']
        x, y, w, h = map(int, bbox)

        # Draw bounding box on the canvas
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    img = Image.fromarray(img)
    return img

