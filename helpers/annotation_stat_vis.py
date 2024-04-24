import pandas as pd
from transformers import BertTokenizer, BertModel
import torch, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats

def plot_stat_dr(annotations, category_df, visualization_type, output_path):

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
        cat = category_ids[categories==category]
        data  = embeddings[categories==category]

        model = IsolationForest(contamination=0.05)
        model.fit(data)
        is_not_outlier = model.predict(data)

        if visualization_type == "hnne":
            from hnne import HNNE
            hnne = HNNE(dim=2)
            projection = hnne.fit_transform(data)
        elif visualization_type == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            projection = tsne.fit_transform(data)  

        scenters_x = projection[:, 0]
        scenters_y = projection[:, 1]
        name = categories_name[categories==category].unique()[0]    
    
        fig, ax = plt.subplots(figsize=(5, 5))

        cmap = plt.cm.get_cmap('Set1')

        point_size=  5
        plt.scatter(scenters_x, scenters_y, 
                label=is_not_outlier, 
                alpha=0.15, 
                color = cmap(is_not_outlier),
                s=point_size)

        # ax.axis('off')  # Turn off axis
        ax.set_xticks([])  # Turn off x-axis ticks
        ax.set_yticks([])
        plt.plot()
        plt.title(name)
        file_names += [f'{output_path}/dr/{name}_data.png']
        os.makedirs(f'{output_path}/dr/', exist_ok=True)
        plt.savefig(f'{output_path}/dr/{name}_data.png')
    
    return file_names
        
def plot_stat_box(annotations, category_df, output_path):
    embeddings = np.concatenate([
        np.stack(annotations["bbox"]), 
        np.expand_dims(np.array(annotations["area"]), 1 ) 
    ], 1)


    new_categories = []
    counter = 0
    mapping = {}
    mapping_name = {}
    remaining_cat =  category_df[category_df['id'].isin(annotations['category_id'].unique())]

    # import pdb; pdb.set_trace()
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

    # Assign a color to each category
    file_names = []

    for category in tqdm(sc):
        cat = category_ids[categories==category]
        data  = embeddings[categories==category]

        model = IsolationForest(contamination=0.05)
        model.fit(data)
        is_not_outlier = model.predict(data)

        name = categories_name[categories==category].unique()[0]    
            
        cmap = plt.cm.get_cmap('Set1')

        point_size=  5
        # Create subplots for each category
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 5), sharey=False)

        # Iterate through each category and create a box plot
        labels=['X', 'Y', 'Width', 'Height', "Area"]
        for i in range(5):
            axs[i].boxplot(data[:, i])
            axs[i].set_title(f'Category {labels[i]}')

        for ax in axs:
            ax.set_xlabel('Values')
        axs[0].set_ylabel('Categories')

        # Adjust layout to prevent clipping of titles
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.title(name)
        plt.plot()
        file_names += [f'{output_path}/box/{name}_box.png']
        os.makedirs(f'{output_path}/box/', exist_ok=True)
        plt.savefig(f'{output_path}/box/{name}_box.png')
    
    return file_names