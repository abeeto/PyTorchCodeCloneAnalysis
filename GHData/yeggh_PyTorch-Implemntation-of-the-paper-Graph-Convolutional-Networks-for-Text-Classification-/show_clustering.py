from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from utils.utils import load_corpus


def visualize_dataset(data_set):
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']

    if data_set not in datasets:
        sys.exit("wrong dataset name")

    _, _, y_train, _, _, _, _, _, _, _ = load_corpus(data_set)
    label_count = y_train.shape[1]

    data_path = './data'

    with open(os.path.join(data_path, data_set + '.train.index'), 'r') as f:
        lines = f.readlines()
        train_size = len(lines)

    with open(os.path.join(data_path, data_set + '_shuffle.txt'), 'r') as f:
        lines = f.readlines()

    target_names = set()
    labels = []
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        labels.append(temp[2])
        target_names.add(temp[2])

    target_names = list(target_names)

    with open(os.path.join(data_path, data_set + '_doc_vectors.txt'), 'r') as f:
        lines = f.readlines()

    docs = []
    for line in lines:
        temp = line.strip().split()
        values_str_list = temp[1:]
        values = [float(x) for x in values_str_list]
        docs.append(values)

    label = labels[train_size:]
    label = np.array(label)

    print("Starting TSNE dimension reduction...")
    transformed = TSNE(n_components=2).fit_transform(docs[train_size:])
    print("Dimension reduction done!")
    pdf = PdfPages(os.path.join('tsne', data_set + '_clustering_output.pdf'))
    class_number = np.unique(label)

    transformed_points = [transformed[label == i] for i in class_number]
    for i, p in enumerate(transformed_points):
        if class_number[i] in range(label_count):
            plt.scatter(p[:, 0], p[:, 1], label=class_number[i], marker='+')
        else:
            plt.scatter(p[:, 0], p[:, 1], label=class_number[i])
    plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.48, -0.08), fontsize=11)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    pdf.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: python show_clustering.py <dataset>")
    data_set = sys.argv[1]
    visualize_dataset(data_set)
