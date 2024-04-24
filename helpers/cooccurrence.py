import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 
import pandas as pd

def plot_cooccurrence(annotations_df, categories_df, output_path):
    file_names = []
    image_ids = annotations_df["image_id"].unique()
    categories = categories_df["name"].unique()

    cooccurrence_matrix  = np.zeros((len(categories), len(categories)))

    for image_id in tqdm(image_ids):
        annotations_per_image = annotations_df[annotations_df["image_id"]==image_id]
        categories_per_image  = annotations_per_image["category_id"].unique()
        for cat1 in categories_per_image:
            for cat2 in categories_per_image:
                cooccurrence_matrix[cat1-1,cat2-1] += 1

    coma_df = pd.DataFrame(cooccurrence_matrix, (categories),  categories)
    coma_df.to_csv(f"{output_path}/cooccurrence_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cooccurrence_matrix/len(image_ids), cmap='viridis')

    # Add color bar
    cbar = fig.colorbar(cax)

    # Set labels for rows and columns
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=90)
    ax.set_yticklabels(categories)

    os.makedirs(os.path.join(output_path,"cooccurrence"), exist_ok=True)
    plt.savefig(os.path.join(output_path,"cooccurrence", 'cooccurrence.png'), bbox_inches='tight')
    file_names += [os.path.join(output_path,"cooccurrence", 'cooccurrence.png')]
    return file_names