import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def get_category_count(coco_all, coco_categories, out_path):
    out = []
    supercategories = list(coco_categories.keys())

    for i, supercategory in enumerate(supercategories):
        sorted_dict = dict(sorted(coco_categories[supercategory].items(), key=lambda item: item[1]))
        for j, category in enumerate(sorted_dict.keys()):
            out += [
                {
                    "supercategory": supercategory,
                    "category": category,
                    "value": sorted_dict[category]
                }
            ]

    pd.DataFrame(out).to_csv(os.path.join(out_path, "instances.csv"))


# Finished - roughly
def plot_histogram_for_categories(coco_all, coco_categories, out_path):
    get_category_count(coco_all, coco_categories, out_path)
    
    cmap = plt.cm.get_cmap('Paired')

    # Assume you have already extracted category_counts as described in the previous responses

    supercategories = list(coco_categories.keys())
    categories = list(set(category for sub_dict in coco_categories.values() for category in sub_dict.keys()))

    fig, ax = plt.subplots(figsize=(20, 8))
    bar_width = 1  # Adjusted bar size

    base = 0
    ticks = []
    for i, supercategory in enumerate(supercategories):
        counts = [coco_categories[supercategory].get(category, 0) for category in categories]
        sorted_dict = dict(sorted(coco_categories[supercategory].items(), key=lambda item: item[1]))

        for j, k in enumerate(sorted_dict.keys()):
            if base<13:
                ax.bar(base, sorted_dict[k], width=bar_width, color=cmap((0 % 8) / 8))
            else:
                ax.bar(base, sorted_dict[k], width=bar_width, color=cmap((i % 8) / 8))
            
            base += 1

            ticks += [k]

    # Add dotted lines at 10, 100, and 1000 on the y-axis
    ax.axhline(y=10, linestyle='--', color='gray', linewidth=0.8)
    ax.axhline(y=100, linestyle='--', color='gray', linewidth=0.8)
    ax.axhline(y=1000, linestyle='--', color='gray', linewidth=0.8)

    ax.set_xticks(np.arange(0,len(categories),1))  # Use this line instead of the original
    # ax.set_xticklabels(categories, rotation=45, fontsize = 6)  # Rotate x-axis ticks by 90 degrees

    # import pdb; pdb.set_trace()
    ax.set_xticklabels(ticks, 
                       rotation=90, fontsize = 12)  # Rotate x-axis ticks by 90 degrees

    # Show y-axis ticks in absolute values
    plt.yscale('log')  # Set y-axis scale to log
    ytick_values = [1, 10, 100, 1000, 10000]  # Define the desired y-axis tick values
    ax.set_yticks(ytick_values)
    # ax.set_yticks([1, 10, 100, 1000])  # Show y axis ticks by absolute values

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())  # Show ticks in absolute values


    ax.set_xlim(-0.5, base)  # Adjust xlim to remove white borders

    ax.tick_params(axis='x', labelsize=12)  # Adjust the font size as needed

    ax.tick_params(axis='y', labelsize=24)  # Adjust the font size as needed

    # ax.legend(supercategories, title='Supercategories', fontsize=12)

    plt.grid()
    # plt.xlabel('Categories')
    plt.ylabel('Number of Instances (log scale)', fontsize = 25)
    plt.title('Number of Instances for Each Class Grouped by Supercategory', fontsize = 25)
    # plt.show()
    plt.savefig(os.path.join(out_path, 'instances.png'), bbox_inches='tight')

    return [os.path.join(out_path, 'instances.png')]