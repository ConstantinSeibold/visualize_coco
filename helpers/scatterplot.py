import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm

def scatter_to_csv(annotations, category_df, out_path):
    counter = 0
    mapping = {}
    mapping_name = {}
    remaining_cat =  category_df[category_df['id'].isin(annotations['category_id'].unique())]

    supercat = {}


    for i in range(len(remaining_cat)):
        cat = remaining_cat.iloc[i]
        if cat["supercategory"] not in supercat.keys():
            supercat[cat["supercategory"]] = counter
            counter += 1
        mapping_name[cat['id']] = cat["supercategory"]
        mapping[cat['id']] = counter 

    category_ids = annotations['category_id']
    mapping2 = {m:mapping[m] for m in mapping}
    supercategories = category_ids.map(mapping2)
    supercategories_name = category_ids.map(mapping_name)
 
    sc = supercategories.unique()

    box = np.stack(annotations['bbox'])
    centers_x = box[:,0] + 0.5 * box[:,2]
    centers_y = box[:,1] + 0.5 * box[:,3]

    # cahnge to do it via image_df
    img_width = np.array([1024] * len(centers_x))
    img_height = np.array([1024] * len(centers_x))

    df = pd.DataFrame([annotations["id"],category_ids, supercategories, centers_x, centers_y, img_width, img_height],
                    ("id","category_id","supercategory_id","center_x","center_y","img_width","img_height")).T
    

    df.to_csv(f'{out_path}/positions.csv')

def plot_scatterplot_for_categories(annotations, category_df, out_path):
    scatter_to_csv(annotations, category_df, out_path)
    counter = 0
    mapping = {}
    mapping_name = {}
    remaining_cat =  category_df[category_df['id'].isin(annotations['category_id'].unique())]

    supercat = {}


    for i in range(len(remaining_cat)):
        cat = remaining_cat.iloc[i]
        if cat["supercategory"] not in supercat.keys():
            supercat[cat["supercategory"]] = counter
            counter += 1
        mapping_name[cat['id']] = cat["supercategory"]
        mapping[cat['id']] = counter 
        

        

    # annotations = coco_all.loadAnns(coco_all.getAnnIds())

    # annotations = pd.DataFrame(annotations)
    category_ids = annotations['category_id']
    mapping2 = {m:mapping[m] for m in mapping}
    supercategories = category_ids.map(mapping2)
    supercategories_name = category_ids.map(mapping_name)
 
    sc = supercategories.unique()
    # import pdb;pdb.set_trace()
    box = np.stack(annotations['bbox'])
    centers_x = box[:,0] + 0.5 * box[:,2]
    centers_y = box[:,1] + 0.5 * box[:,3]


    # Assign a color to each supercategory
    file_names = []

    for supercategory in sc:
        cat = category_ids[supercategories==supercategory]
        scenters_x = centers_x[supercategories==supercategory]
        scenters_y = centers_y[supercategories==supercategory]
        name = supercategories_name[supercategories==supercategory].unique()[0]    

        # color = supercategory_colors[supercategory]
        
        fig, ax = plt.subplots(figsize=(5, 5))

        cmap = plt.cm.get_cmap('Set1', len(cat.unique()))
        
        # implot = plt.imshow(out_img, cmap=plt.cm.get_cmap('binary'))
        
        point_size=  5
        plt.scatter(scenters_x, scenters_y, 
                # label=cat, 
                alpha=0.15, 
                color = (cat%len(cat.unique())/len(cat.unique())).apply(cmap),
                cmap = cmap,
                s=point_size)

        # ax.axis('off')  # Turn off axis
        ax.set_xticks([])  # Turn off x-axis ticks
        ax.set_yticks([])
        
        plt.gca().invert_yaxis()
        plt.plot()
        plt.savefig(f'{out_path}/{name}_position.png')
        file_names += [f'{out_path}/{name}_position.png']

    return file_names
        

    # ax.legend()
    # plt.xlabel('Center X Coordinate')
    # plt.ylabel('Center Y Coordinate')
    # plt.title('Scatter Plot of Instance Centers by Supercategory')
    # plt.show()
