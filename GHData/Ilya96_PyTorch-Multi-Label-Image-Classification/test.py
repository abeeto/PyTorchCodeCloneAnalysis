import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset, FashionDataset, mean, std
from model import MultiOutputModel, visualize_gt_data, calculate_metrics, validate

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


def visualize_grid(dataset, attributes, color_predictions, gender_predictions, article_predictions):
    imgs = []
    labels = []
    predicted_color_all = []
    predicted_gender_all = []
    predicted_article_all = []

    gt_labels = []
    gt_color_all = []
    gt_gender_all = []
    gt_article_all = []

    # store the original transforms from the dataset
    transforms = dataset.transform
    # and not use them during visualization
    dataset.transform = None

    for (sample,
         predicted_color,
         predicted_gender,
         predicted_article) in zip(
        dataset, color_predictions, gender_predictions, article_predictions):
        predicted_color = attributes.color_id_to_name[predicted_color]
        predicted_gender = attributes.gender_id_to_name[predicted_gender]
        predicted_article = attributes.article_id_to_name[predicted_article]

        gt_color = attributes.color_id_to_name[sample['labels']['color_labels']]
        gt_gender = attributes.gender_id_to_name[sample['labels']['gender_labels']]
        gt_article = attributes.article_id_to_name[sample['labels']['article_labels']]

        predicted_color_all.append(predicted_color)
        predicted_gender_all.append(predicted_gender)
        predicted_article_all.append(predicted_article)

        gt_color_all.append(gt_color)
        gt_gender_all.append(gt_gender)
        gt_article_all.append(gt_article)

        imgs.append(sample['img'])
        labels.append("{}\n{}\n{}".format(predicted_gender, predicted_article, predicted_color))
        gt_labels.append("{}\n{}\n{}".format(gt_gender, gt_article, gt_color))

    # restore original transforms
    dataset.transform = transforms

    # Draw confusion matrices
    # color
    cn_matrix = confusion_matrix(
        y_true=gt_color_all,
        y_pred=predicted_color_all,
        labels=attributes.color_labels,
        normalize='true')

    plt.rcParams.update({'font.size': 5})
    plt.rcParams.update({'figure.dpi': 300})
    ConfusionMatrixDisplay(cn_matrix, attributes.color_labels).plot(
        include_values=False, xticks_rotation='vertical')
    plt.title("Colors")
    plt.tight_layout()
    plt.show()

    # gender
    cn_matrix = confusion_matrix(
        y_true=gt_gender_all,
        y_pred=predicted_gender_all,
        labels=attributes.gender_labels,
        normalize='true')
    ConfusionMatrixDisplay(cn_matrix, attributes.gender_labels).plot(
        xticks_rotation='horizontal')
    plt.title("Genders")
    plt.tight_layout()
    plt.show()

    plt.rcParams.update({'font.size': 2.5})
    cn_matrix = confusion_matrix(
        y_true=gt_article_all,
        y_pred=predicted_article_all,
        labels=attributes.article_labels,
        normalize='true')
    ConfusionMatrixDisplay(cn_matrix, attributes.article_labels).plot(
        include_values=False, xticks_rotation='vertical')
    plt.title("Article types")
    plt.show()

    plt.rcParams.update({'font.size': 5})
    plt.rcParams.update({'figure.dpi': 100})
    title = "Predicted labels"
    n_cols = 5
    n_rows = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.flatten()
    for img, ax, label in zip(imgs, axs, labels):
        ax.set_xlabel(label, rotation=0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def test(checkpoint_path):
    attributes_file = 'fashion-product-images/styles.csv'

    device = torch.device("cuda")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('fashion-product-images/val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = MultiOutputModel(n_color_classes=attributes.num_colors, n_gender_classes=attributes.num_genders,
                             n_article_classes=attributes.num_articles).to(device)

    model_predictions = validate(model, test_dataloader, device, checkpoint=checkpoint_path)

    # Visualization of the trained model
    visualize_grid(test_dataset, attributes, *model_predictions)


last_checkpoint_path = "VGG16_200.pth"
test(last_checkpoint_path)
