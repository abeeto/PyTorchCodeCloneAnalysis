#!/usr/bin/env python

import os
import json
import time
import random
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.misc import imsave

# from cnn import eucl_dist_output_shape, contrastive_loss, euclidean_distance
from dataset_torch import DATASET_NPY
from loop_torch import test_loop, train_model
from model_torch import ICONNet
import settings
import utils
import torch
from torch import device, nn
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split, Subset
from PIL import Image

# def euc_dist(x):
#     'Merge function: euclidean_distance(u,v)'
#     s = x[0] - x[1]
#     output = (s ** 2).sum(axis=1)
#     output = K.reshape(output, (output.shape[0],1))
#     return output

# def euc_dist_shape(input_shape):
#     'Merge output shape'
#     shape = list(input_shape)
#     outshape = (shape[0][0],1)
#     return tuple(outshape)

def plot_model_history(model_history, filename):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1),
                model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1),
                model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,
                                len(model_history.history['acc']) + 1),
                      len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,
                                len(model_history.history['loss']) + 1),
                      len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(filename)
    plt.close(fig)


def accuracy(dataset: DATASET_NPY, model: ICONNet, class_names, cm_filename, datagen, words=None, anomaly=False):
    num_classes = len(class_names)
    if words is not None:
        # x, y = word_datagen(datagen, x, words, y, len(x)).next()
        print('words', words)
    else:
        # x, y = datagen.flow(x, y, batch_size=len(x), shuffle=False).next()
        # print(dataset.data.numpy()[0].reshape(32,32)/255.0)
        x = DataLoader(dataset, batch_size=32, # dataset.train_data.shape[0]
                            num_workers=8, pin_memory=True)
        # inputs, classes = next(iter(x)) # DONE:debug
        # print(inputs.numpy()[0].reshape(32,32)) # debug
    result = test_loop(x, model, nn.CrossEntropyLoss())
    
    x = dataset.data.numpy()
    y = dataset.targets.numpy()
    y = utils.to_categorical(y, num_classes)
    # print(len(set(np.argmax(y, axis=1))))
    result = result.cpu().numpy()

    anomaly_class = num_classes
    anomalies = np.zeros(len(result))
    if anomaly:
        from dl_inference_service import DlInferenceService
        dlis = DlInferenceService()
        anomalies = dlis.anomaly_model.predict(result)
        class_names += ['anomaly']
        num_classes += 1
    predicted_class = np.argmax(result, axis=1)
    predicted_class[anomalies == 1] = anomaly_class
    true_class = np.argmax(y, axis=1)
    if anomaly:
        predicted_class[0] = anomaly_class
        true_class[0] = anomaly_class
    # print(len(set(true_class)))
    print(predicted_class[0:10])
    print(true_class[0:10])
    np.save('saved_models/small.npy', x[0:10])
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]

    # draw the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for idx in range(len(predicted_class)):
        confusion_matrix[true_class[idx]][predicted_class[idx]] += 1

    conf_arr = confusion_matrix
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i)
        for j in i:
            try:
                tmp_arr.append(float(j) / float(a))
            except ZeroDivisionError:
                tmp_arr.append(0)
        # the sqrt makes the colors a bit better
        norm_conf.append(np.sqrt(tmp_arr))

    fig = plt.figure(figsize=(40, 40), dpi=150)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for xx in range(width):
        for yy in range(height):
            ax.annotate(str(int(conf_arr[xx][yy])), xy=(yy, xx),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=10)

    # cb = fig.colorbar(res)
    plt.xticks(range(width), class_names, rotation=90)
    plt.yticks(range(height), class_names)
    plt.savefig(cm_filename)
    print(cm_filename)

    return accuracy * 100, predicted_class, true_class, x


def load_data(data_folder):
    x_train = np.load(data_folder("training_x.npy"))
    y_train = np.load(data_folder("training_y.npy"))
    x_test = np.load(data_folder("validation_x.npy"))
    y_test = np.load(data_folder("validation_y.npy"))

    # Print Training and Testing data dimension
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Pre-process data, so value is between 0.0 and 1.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    num_classes = np.unique(y_train).shape[0]
    # Print Unique Icon Classes, 99 classes
    # print(np.unique(y_train))
    # print(num_classes, ' classes')

    # Convert class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes


def load_word_data(data_folder):
    x_train = np.load(data_folder("training_x_words.npy"))
    x_test = np.load(data_folder("validation_x_words.npy"))

    return x_train, x_test


def create_model(embedding_size, num_classes, model_type='', siamese=False, conv_only=False):
    return ICONNet(embedding_size, num_classes, model_type=model_type, siamese=siamese, conv_only=conv_only)


def initialize_model(embedding_size, num_classes, model_type=''):
    # p_ratio = [1.0, 1.44, 1.73, 1.0]
    # fmp = Lambda(lambda x: tf.nn.fractional_max_pool(x, p_ratio)[0])

    model = create_model(embedding_size, num_classes, model_type=model_type)

    return model



def initialize_datagen(root, batch_size_train):
    print('Using real-time data augmentation.')
    datagen = {}
    dataset = {}
    types = ['train', 'val']

    ffffff = DATASET_NPY(
        root=root,
        train=True,
        transform=transforms.ToTensor(), # Pre-process data, so value is between 0.0 and 1.0
    )
    # print(ffffff.data.numpy()[0].reshape(32,32)/255.0)
    x = DataLoader(ffffff, batch_size=len(ffffff), # dataset.train_data.shape[0]
                            num_workers=8, pin_memory=True)
    inputs, classes = next(iter(x)) # debug
    # print(inputs.numpy()[0].reshape(32,32))
    ffffff.fit(inputs)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([ffffff.mean[0][0][0]], [ffffff.std[0][0][0]]),
            transforms.LinearTransformation(ffffff.principal_components, ffffff.mean_vector),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([ffffff.mean[0][0][0]], [ffffff.std[0][0][0]]),
            transforms.LinearTransformation(ffffff.principal_components, ffffff.mean_vector),
        ]),
    } # preprocess
    dataset = {x: DATASET_NPY(root, train=(x=='train'), transform=data_transforms[x])
                        for x in types}
    with open(os.path.join(settings.data_folder("validation_metadata.json")), 'r') as infile:
        class_names = json.load(infile)["class_names"]
    for x in types:
        dataset[x].classes = class_names

    batch_size_ = {
        'train': batch_size_train,
        'val': batch_size_train, # dataset['val'].train_data.shape[0],
    }
    datagen = {x: DataLoader(dataset[x], batch_size=batch_size_[x], shuffle=True,
                            num_workers=2, pin_memory=True) # sampler=self.sampler[x]
            for x in types}

    num_classes = len(dataset['train'].classes)
    return datagen, dataset, num_classes, data_transforms['val']


# def word_datagen(datagen, x, x_words, y, batch_size):
#     generator = ImageDataGenerator()
#     normal = datagen.flow(x, y, shuffle=False, batch_size=batch_size)
#     original_shape = x_words.shape
#     x2 = generator.flow(x_words.reshape(original_shape + (1, 1)), shuffle=False, batch_size=batch_size)
#     while True:
#         x1, y1 = normal.next()
#         words = x2.next()
#         if words.shape[0] != batch_size:
#             print(words.shape)
#             continue
#         yield [x1, words.reshape((batch_size, original_shape[1]))], y1


def make_buckets(x, y):
    ys = np.unique(y)
    print(ys)
    y = y.flatten()
    buckets = {int(c): x[y == c] for c in ys}
    for bucket in buckets:
        print(buckets[bucket].shape)
    return buckets


def pick_example(buckets, positive, classification):
    if positive:
        return random.choice(buckets[classification])
    options = buckets.keys()
    options.remove(classification)
    return random.choice(buckets[random.choice(options)])
    


def save_model(model, save_dir, epochs, embedding_size):
    model_path = os.path.join(save_dir, "small_cnn_weights_{}_{}.pt".format(epochs, embedding_size))
    torch.save(model, model_path)
    # model_json_path = os.path.join(save_dir, 'small_cnn_{}.json'.format(embedding_size))

    # with open(model_json_path, 'w') as outfile:
    #     outfile.write(model.to_json())

    print('Saved trained model at {}'.format(model_path))


def save_confusion_matrix(data_type, datagen, model, dataset: DATASET_NPY, class_names, cnn_params, words=None, anomaly=False, exp_name=''):
    """:param data_type: string representing 'training' or 'test'"""
    embedding_size = cnn_params["embedding_size"]
    epochs = cnn_params["epochs"]
    save_dir = cnn_params["save_dir"]

    file_path = os.path.join(save_dir, 'confusion_{}_{}_{}.png'.format(data_type,
                                                                       epochs,
                                                                       embedding_size))
    acc, y_pred, y_true, datagen_x = accuracy(dataset,
                                              model,
                                              class_names,
                                              file_path,
                                              datagen,
                                              words,
                                              anomaly)

    results = [
        "Accuracy on {} data is: {:0.2f}".format(data_type, acc),
        "Macro precision",
        metrics.precision_score(y_true, y_pred, average='macro'),
        # "Micro precision",
        # metrics.precision_score(y_true, y_pred, average='micro'),
        "Macro recall",
        metrics.recall_score(y_true, y_pred, average='macro'),
        # "Micro recall",
        # metrics.recall_score(y_true, y_pred, average='micro'),
    ]

    results_string = "\n".join(str(result) for result in results)
    print(results_string)
    with open(os.path.join(save_dir, 'results_{}.txt'.format(data_type)), 'w') as results_file:
        results_file.write(results_string)

    if exp_name:
        x = dataset.data.numpy()
        metadata = write_images(x, y_pred, y_true, class_names, data_type, save_dir, exp_name)
        with open(os.path.join(save_dir, "images_{}{}.json".format(data_type, exp_name)), "w") as metadata_file:
            json.dump(metadata, metadata_file)


def write_images(x, y, y_true, class_names, data_type, save_dir, exp_name=''):
    path = settings.PATHS['icons']

    metadata = {
        "data": {},
        "metadata": {
            "type": "classification_result",
            "subfolder": data_type,
            "exp": exp_name or save_dir,
        },
    }

    precisions = {label: metrics.precision_score(y_true, y, average='micro', labels=[idx])
                  for idx, label in enumerate(class_names)}
    recalls = {label: metrics.recall_score(y_true, y, average='micro', labels=[idx])
               for idx, label in enumerate(class_names)}

    for idx, row in enumerate(x):
        image = row.reshape(32, 32)
        class_name = class_names[int(y[idx])]
        image_folder = os.path.join(path, metadata["metadata"]["exp"], data_type, class_name)
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "{}.png".format(idx))
        imsave(image_path, image)
        metadata["data"][class_name] = metadata["data"].get(class_name,
                                                            {
                                                                "resource_id": class_name,
                                                                "top_words": [],
                                                                "closest_training_icons": [],
                                                                "precision": precisions[class_name],
                                                                "recall": recalls[class_name],
                                                            })
        metadata["data"][class_name]["closest_training_icons"] += [idx]

    return metadata


# def generate_embeddings(model, x_train, y_train, data_type, save_dir, datagen):
#     embedding_model = Sequential()
#     for layer in model.layers:
#         embedding_model.add(layer)
#         if layer.name == 'embedding':
#             break

#     x = datagen.flow(x_train, batch_size=len(x_train), shuffle=False).next()
#     embeddings = embedding_model.predict(x_train)
#     print(embeddings.shape)
#     np.save("validation_x_embedding.npy", embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",# default='saved_models/pyt_model.pt',
                        help=("Full path to the h5 model file"))
    parser.add_argument("--save_images",
                        help="Where to save the test image classifications")
    parser.add_argument("--siamese", action='store_true', help='Siamese or not?')
    parser.add_argument("--words", action='store_true', help='Words or not?')
    parser.add_argument("--visualize", help='Want to just visualize the model? Put a path to where you want the image!')
    parser.add_argument("--model_type", help='Simple or not?')
    parser.add_argument("--save_dir", help='Subfolder to save images in')
    parser.add_argument("--embeddings", action='store_true', help='Generate embeddings?')
    parser.add_argument("--anomaly", action='store_true', help='Measure accuracy with anomaly detection')
    parser.add_argument("-i","--input_img",# default='29207.png'
                        )
    args = parser.parse_args()

    np.random.seed(settings.RANDOM_SEED)
    random.seed(settings.RANDOM_SEED)
    cnn_params = settings.CNN_PARAMS
    embedding_size = cnn_params["embedding_size"]
    batch_size = cnn_params["batch_size"]
    epochs = cnn_params["epochs"]
    save_dir = args.save_dir or cnn_params["save_dir"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # x_train, x_test, y_train, y_test, num_classes = load_data(settings.data_folder)

    train_words = test_words = None
    if args.words:
        train_words, test_words = load_word_data(settings.data_folder)

    datagen, dataset, num_classes, data_transform = initialize_datagen(settings.data_folder_path, batch_size)

    # with open(os.path.join(save_dir, 'datagen.pkl'), 'wb') as datagen_file:
    #     pickle.dump(datagen, datagen_file)

    if not args.model_path:
        model = initialize_model(cnn_params["embedding_size"], num_classes, args.model_type)
        model.load_state_dict(torch.load('saved_models_0/pyt_model_state_dict.pt'))
        # opt = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-4)
        # loss= nn.CrossEntropyLoss() # 'categorical_crossentropy'
        # # metrics=['accuracy']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        start_time = time.time()
        model_info = train_model(model, datagen, epochs, device=device)
        end_time = time.time()

        print("Model took %0.2f minutes to train" % ((end_time - start_time) / 60.0))
    else:
        # Model class must be defined somewhere
        model = torch.load(args.model_path) # not like keras, must have the model difination
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

    if args.embeddings:
        # generate_embeddings(model, x_train, y_train, 'train', save_dir, datagen)
        return

    if not args.model_path:
        model_history_file_path = os.path.join(save_dir, "small_cnn_info_{}_{}.pdf".format(epochs,
                                                                                           embedding_size))
        # plot_model_history(model_info, model_history_file_path)

        save_model(model, save_dir, epochs, embedding_size)

    if args.visualize:
        # plot_model(model, to_file=args.visualize, show_shapes=True)
        return

    with open(os.path.join(settings.data_folder("validation_metadata.json")), 'r') as infile:
        class_names = json.load(infile)["class_names"]

    if args.input_img and os.path.isfile(args.input_img):
        img = Image.open(args.input_img)
        img_ = data_transform(img)
        img.show()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            pred = model(torch.unsqueeze(img_, 0).to(device))
        result = nn.Softmax(dim=1)(pred)
        result = result.cpu().numpy()
        predicted_class = np.argmax(result, axis=1)
        print(predicted_class[0], class_names[predicted_class[0]])
        return

    # save_confusion_matrix("train", datagen, model, dataset['train'], class_names, cnn_params, train_words, args.anomaly, args.save_images)
    print()
    save_confusion_matrix("test", datagen, model, dataset['val'], class_names, cnn_params, test_words, args.anomaly, args.save_images)


if __name__ == "__main__":
    main()
