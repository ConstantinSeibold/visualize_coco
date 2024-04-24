import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, json
import pandas as pd
from pycocotools.coco import COCO
from helpers.histogram import plot_histogram_for_categories
from helpers.scatterplot import plot_scatterplot_for_categories
from helpers.category_name_vis import plot_class_similarity, plot_wordcloud, plot_category_dendrogram
from helpers.annotation_stat_vis import plot_stat_dr, plot_stat_box
from helpers.heatmaps import plot_class_heatmaps
from helpers.mean_image import plot_mean_image
from helpers.cooccurrence import plot_cooccurrence
from helpers.plot_images_in_bulk import plot_images_in_dr, plot_images_out_of_dr
# Create a PdfPages object to save multiple pages in the PDF file
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from PIL import Image
import io


def add_png_page(pdf_writer, image_filename):
    # Add a page from a PNG file to the PDF
    img = Image.open(image_filename)
    img = img.convert('RGB')

    pdf_bytes = io.BytesIO()
    img.save(pdf_bytes, format='PDF')

    pdf_bytes.seek(0)
    pdf_reader = PdfReader(pdf_bytes)
    pdf_writer.add_page(pdf_reader.pages[0])

def add_pdf_page(pdf_writer, pdf_filename):
    # Add a page from a PDF file to the PDF
    with open(pdf_filename, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        pdf_writer.add_page(pdf_reader.pages[0])

def generate_pdf(file_path, dataset_name, file_dict):
    writer = PdfWriter()
    writer.add_page(create_title_page(f"Report of {dataset_name}"))
    writer.add_outline_item("Title Page", 0)

    page_num = 1
    for key in file_dict.keys():
        print(key)
        print("-------------------")
        writer.add_page(create_section_page(key))
        parent_bookmark = writer.add_outline_item(key, page_num)
        page_num += 1

        for path in file_dict[key]:
            if path[-4:] == ".pdf":
                add_pdf_page(writer, path)
            else:
                add_png_page(writer, path)
            writer.add_outline_item(path.split("/")[-1][:-4], page_num, parent_bookmark)
            page_num += 1

    writer.write(file_path)

def create_title_page(title):
    # Create a title page for the PDF
    packet = io.BytesIO()
    can = canvas.Canvas(packet)
    can.setFont("Helvetica", 24)
    can.drawString(100, 500, title)
    can.save()

    packet.seek(0)
    pdf_reader = PdfReader(packet)
    return pdf_reader.pages[0]

def create_section_page(section_title):
    # Create a page for a section title
    packet = io.BytesIO()
    can = canvas.Canvas(packet)
    can.setFont("Helvetica", 16)
    can.drawString(100, 500, section_title)
    can.save()

    packet.seek(0)
    pdf_reader = PdfReader(packet)
    return pdf_reader.pages[0]


def get_dfs(annotation_path):

    assert os.path.isfile(annotation_path)

    lmao= json.load(open(annotation_path))
    ano = pd.DataFrame(lmao['annotations'])
    img = pd.DataFrame(lmao['images'])
    categories = pd.DataFrame(lmao['categories'])
    if "supercategory" not in categories.keys():
        categories["supercategory"] = "None"
 
    return ano, img, categories

def get_coco(json_path):
    coco = COCO(json_path)

    categories = coco.loadCats(coco.getCatIds())
    category_counts = {}

    for category in categories:
        if "supercategory" not in category.keys():
            supercategory = "None"
        else:
            supercategory = category['supercategory']
        category_name = category['name']
        category_id = category['id']
        instances = len(coco.getImgIds(catIds=[category_id]))
        
        if supercategory not in category_counts:
            category_counts[supercategory] = {}

        category_counts[supercategory][category_name] = instances
    
    return coco, category_counts

def visualize(input_path, output_path, image_root, visualization_type):
    # Your main logic goes here
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")

    os.makedirs(output_path, exist_ok=True)
    out_path_indv = os.path.join(output_path, "indv_images")
    os.makedirs(os.path.join(output_path, "indv_images"), exist_ok=True)

    ano, img, categories = get_dfs(input_path)
    coco, coco_category_counts = get_coco(input_path)
    
    # with PdfPages(os.path.join(output_path, 'multipage_plot.pdf')) as pdf:
    #    pdf.savefig()

    print(f"Visualization Type: {visualization_type}")
    if visualization_type == "histogram":
        plot_histogram_for_categories(coco, coco_category_counts, out_path_indv)
    elif visualization_type == "scatterplot":
        plot_scatterplot_for_categories(ano, categories, out_path_indv)
    elif visualization_type == "class_sim":
        plot_class_similarity(categories, "hnne", out_path_indv)
    elif visualization_type == "feats":
        plot_stat_dr(ano, categories, "hnne", out_path_indv)
    elif visualization_type == "box":
        plot_stat_box(ano, categories, out_path_indv)
    elif visualization_type == "wordcloud":
        plot_wordcloud(categories, out_path_indv)
    elif visualization_type == "dendrogram":
        #Todo
        plot_category_dendrogram(categories, out_path_indv)
    elif visualization_type == "heatmap":
        plot_class_heatmaps(img, ano, categories, out_path_indv)
    elif visualization_type == "cooccurrence":
        # Todo chord diagram
        plot_cooccurrence(ano, categories, out_path_indv)
    elif visualization_type == "mean":
        # Todo mean
        plot_mean_image(img, image_root, out_path_indv)
    elif visualization_type == "samples":
        plot_images_in_dr(ano, img, categories, image_root, out_path_indv)
    elif visualization_type == "outlier":
        plot_images_out_of_dr(ano, img, categories, image_root, out_path_indv)
    elif visualization_type == "report":

        subfiles = {
            "histogram": plot_histogram_for_categories(coco, coco_category_counts, out_path_indv),
            "scatterplot": plot_scatterplot_for_categories(ano, categories, out_path_indv),
            "class_sim": plot_class_similarity(categories, "hnne", out_path_indv),
            "feats": plot_stat_dr(ano, categories, "tsne", out_path_indv),
            "box": plot_stat_box(ano, categories, out_path_indv),
            "wordcloud": plot_wordcloud(categories, out_path_indv),
            "cooccurrence": plot_cooccurrence(ano, categories, out_path_indv),
            "dendrogram": plot_category_dendrogram(categories, out_path_indv),
            "heatmap": plot_class_heatmaps(img, ano, categories, out_path_indv),
            "mean": plot_mean_image(img, image_root, out_path_indv),
            "samples": plot_images_in_dr(ano, img, categories, image_root, out_path_indv),
            "outlier": plot_images_out_of_dr(ano, img, categories, image_root, out_path_indv),
        }
        # Todo total report
        pdf_file_path = os.path.join(output_path,"report.pdf")
        generate_pdf(pdf_file_path, input_path.split("/")[-1],subfiles)
        


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Script description")

    # Add arguments
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("--output_path",default="testing", help="Path to output file")
    parser.add_argument("--image_root",default="testing", help="Path to output file")
    parser.add_argument(
        "--vis",
        # choices=["type1", "type2", "type3"],
        default="report",
        help="Type of visualization (type1, type2, type3)",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    visualize(args.input_path, args.output_path, args.image_root, args.vis)
