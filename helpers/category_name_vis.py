import pandas as pd
from transformers import BertTokenizer, BertModel
import torch, os
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over the tokens
    return embeddings

def plot_category_dendrogram(categories_df, output_path):
    if False:
        label_encoder = LabelEncoder()
        categories_df['supercategory'] = label_encoder.fit_transform(categories_df['supercategory'])

        # Creating a linkage matrix using hierarchical clustering
        linked = linkage(categories_df[['supercategory']], method='ward')

        # Plotting dendrogram
        dendrogram(linked, orientation='top', labels=categories_df['name'].tolist(),
                distance_sort='descending', show_leaf_counts=True,
                )
        plt.title('Dendrogram of Categories as Children Nodes of Supercategories')
        plt.xlabel('Categories')
        plt.ylabel('Distance')

        os.makedirs(os.path.join(output_path, "dendrogram"), exist_ok=True)
        plt.savefig(os.path.join(output_path, "dendrogram", 'dendrogram.png'), bbox_inches='tight')
    
    # Create a hierarchical DataFrame
    plt.figure(figsize=(10, 5))
    plt.axis('off')  # Turn off axis labels
    
    G = nx.Graph()
    
    G.add_edges_from(
        [
            ("s_"+categories_df.iloc[i]["supercategory"], categories_df.iloc[i]["name"]) for i in range(len(categories_df))
        ]
    )
    G.add_edges_from(
        [
            (0, "s_"+categories_df.iloc[i]["supercategory"]) for i in range(len(categories_df))
        ]
    )
    # Calculate the positions for a radial tree layout using graphviz_layout
    pos = graphviz_layout(G, prog="twopi", root=0)

    # Draw the tree
    nx.draw(G, pos, with_labels=True, node_size=20, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", font_family="sans-serif")

    os.makedirs(os.path.join(output_path, "dendrogram"), exist_ok=True)
    plt.savefig(os.path.join(output_path, "dendrogram", 'dendrogram.png'), bbox_inches='tight')

    return [os.path.join(output_path, "dendrogram", 'dendrogram.png')]
    # Display the plot

def plot_wordcloud(category_df, output_path):
    text_data = " ".join(category_df["name"].tolist())

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis labels
    os.makedirs(os.path.join(output_path,"wordcloud"), exist_ok=True)
    plt.savefig(os.path.join(output_path,"wordcloud", 'wordcloud.png'), bbox_inches='tight')

    return [os.path.join(output_path,"wordcloud", 'wordcloud.png')]

def plot_class_similarity(category_df, visualization_type, output_path):
    embeddings = extract_text_embeddings_from_coco_json(category_df)
    embeddings = torch.cat(embeddings,0).detach().numpy()


    if visualization_type == "hnne":
        from hnne import HNNE
        hnne = HNNE(dim=2)
        projection = hnne.fit_transform(embeddings)
    elif visualization_type == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        projection = tsne.fit_transform(embeddings)  

    sc_mapping = {value:key for key,value in category_df["supercategory"].to_dict().items()}
    supercategories = category_df["supercategory"].map(sc_mapping)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(projection[:, 0], projection[:, 1], c=supercategories, cmap='viridis')

    unique_labels = np.unique(supercategories)
    for label in unique_labels:
        indices = np.where(supercategories == label)[0]
        center_x = np.mean(projection[indices, 0])
        center_y = np.mean(projection[indices, 1]) - 2
        plt.text(center_x, center_y, category_df["supercategory"].iloc[label], fontsize=12, color='black',
                ha='center', va='center', fontweight='bold')
    
    # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot with Labels')
    plt.colorbar(label='Labels')
    
    os.makedirs(os.path.join(output_path,"class_scatter"), exist_ok=True)
    plt.savefig(os.path.join(output_path,"class_scatter", 'class_name_similarity.png'), bbox_inches='tight')

    return [os.path.join(output_path,"class_scatter", 'class_name_similarity.png')]
    # import pdb;pdb.set_trace()

def extract_text_embeddings_from_coco_json(category_df):

    # Extract class names from the JSON file
    class_names = category_df['name'].tolist()

    # Extract text embeddings for each class name
    class_embeddings = [get_text_embedding(class_name) for class_name in class_names]

    return class_embeddings
