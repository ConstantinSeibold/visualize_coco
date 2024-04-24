import json
from PIL import Image, ImageDraw
import numpy as np
from copy import deepcopy
from pycocotools import mask as coco_mask
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def create_heatmap_image(numpy_array, colormap="plasma"):
    """
    Create a heatmap image from a numpy array.

    Args:
        numpy_array (np.ndarray): Input 2D numpy array.
        colormap (str): Colormap name for visualization.

    Returns:
        Image: Pillow Image object representing the heatmap.
    """
    # Normalize the array values to be in the range [0, 1]
    normalized_array = (numpy_array - np.min(numpy_array)) / (np.max(numpy_array) - np.min(numpy_array))

    # Convert the normalized array to RGB values using a colormap
    colors = plt.cm.get_cmap(colormap)(normalized_array)

    # Convert the RGB values to Pillow image format
    image_array = (colors[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(image_array)

    return image

def rle_to_binary_mask(rle_mask):
    """
    Convert RLE-encoded mask to binary mask.

    Args:
        rle_mask (dict): RLE-encoded mask dictionary.

    Returns:
        np.ndarray: Binary mask.
    """
    rle_mask_copy = deepcopy(rle_mask)
    rle_mask_copy['counts'] = rle_mask_copy['counts']  # .encode('utf-8')
    binary_mask = coco_mask.decode(rle_mask_copy)
    return binary_mask

def plot_class_heatmaps(images_df, annotation_df, category_df, output_path):
    cmap = "plasma"

    image_size = images_df.iloc[0]['height'], images_df.iloc[0]['width']

    class_heatmaps = {}
    class_names = {}
    for i in range(len(category_df)):
        category_info = category_df.iloc[i]
        # import pdb; pdb.set_trace()
        class_id = category_info['id']
        class_name = category_info['name']
        class_heatmaps[category_info['id']] = np.zeros(image_size)
        class_names[class_id] = class_name

    for ii in tqdm(range(len(annotation_df))):
        annotation = annotation_df.iloc[ii]
        if 'segmentation' in annotation:
            category_id = annotation['category_id']

            if 'counts' in annotation['segmentation']:
                # RLE format
                segmentation = rle_to_binary_mask(annotation['segmentation'])
            elif 'segmentation' in annotation:
                # Polygon format
                segmentation = annotation['segmentation']
                # Convert polygon to mask
                rle_mask = coco_mask.frPyObjects(segmentation, image_size[0], image_size[1])
                segmentation = rle_to_binary_mask(rle_mask[0])
            else:
                print("not recognized")
                continue  # Skip if the segmentation format is not recognized

            class_heatmaps[category_id] += segmentation

    for class_name in class_heatmaps.keys():
        class_heatmaps[class_name] = create_heatmap_image(class_heatmaps[class_name], cmap)

    class_heatmaps = {class_names[class_id]: class_heatmaps[class_id] for class_id in class_heatmaps.keys()}

    out_json = []

    file_names = []
    os.makedirs(f'{output_path}/heatmaps', exist_ok=True)
    for class_name, heatmap in class_heatmaps.items():
        heatmap.save(f'{output_path}/heatmaps/{class_name}_heatmap.png')
        file_names += [f'{output_path}/heatmaps/{class_name}_heatmap.png']

        # todo supercategory
        out_json += [
            {
                "category_name": f"{class_name}",
                "supercategory_name": "",
                "filePaths": f'{class_name}_heatmap.png'
            }
        ]

    with open(f"{output_path}/heatmaps.json", "w") as outfile:
        json.dump(out_json, outfile)
    return file_names