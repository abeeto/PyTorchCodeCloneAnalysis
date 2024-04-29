import argparse
import datetime
import json
import os
import pathlib
import random
import shutil
import time

import onnxruntime as ort
import torch
from torch import nn
from torch import optim

import models
from utils import extract_face, extract_labels, write_batch, BmiDataset


def main(args):
    dataset_path = args.train_dataset
    val_dataset_path = args.val_dataset
    data_out = args.data_out
    batch_size = args.batch_size
    seed = args.seed
    epochs = args.epochs

    random.seed(seed)

    # Prepare Training and Validation datasets.
    train_out = os.path.join(data_out, 'train')
    val_out = os.path.join(data_out, 'val')

    if not os.path.exists(train_out):
        load_data(dataset_path, train_out, batch_size)
    if not os.path.exists(val_out):
        load_data(val_dataset_path, val_out, batch_size)

    training_data = BmiDataset(train_out)
    val_data = BmiDataset(val_out)

    # Train and validate model.
    model = train(training_data, epochs)
    evaluation = evaluate(model, val_data)

    # Save experiment data.
    save_experiment(model, evaluation, args)


def train(training_data, epochs):
    model = models.SimpleModel()
    bmi_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for X, bmi in training_data:
            optimizer.zero_grad()

            bmi_out = model(X)
            loss = bmi_criterion(bmi_out, bmi)

            loss.backward()
            optimizer.step()

            print("Loss: %.5f" % (loss.item(),))

    return model


def evaluate(model, val_data):
    in_shape = None
    out_shape = None
    total_count = 0
    total_diff = 0.0

    model.eval()

    start = time.time()
    for X, bmi in val_data:
        if in_shape is None or out_shape is None:
            in_shape = X.shape
            out_shape = bmi.shape

        pred = model(X)
        total_diff += torch.sum((pred - bmi)**2).item()
        total_count += in_shape[0]
    elapsed = time.time() - start

    return {
        'mse': str((total_diff / total_count)),
        'speed_ms': str((elapsed / total_count) * 1000),
        'in_shape': in_shape,
        'out_shape': out_shape
    }


def save_experiment(model, evaluation, config):
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    exp_dir = os.path.join('experiments', config.name)

    if os.path.exists(exp_dir):
        exp_dir = exp_dir + '-' + datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
    os.mkdir(exp_dir)

    cpu = torch.device('cpu')
    model.to(cpu)
    model.eval()

    shape = [size for size in evaluation['in_shape']]
    shape[0] = 1
    dummy_input = torch.randn(shape)

    traced = torch.jit.trace(model, (dummy_input,))

    traced.save(os.path.join(exp_dir, 'model.pt'))

    eval_json = json.dumps(evaluation, sort_keys=True, indent=4)
    with open(os.path.join(exp_dir, 'eval.json'), 'w') as f:
        f.write(eval_json)

    config_json = json.dumps(vars(config), sort_keys=True, indent=4)
    with open(os.path.join(exp_dir, 'conf.json'), 'w') as f:
        f.write(config_json)

    print('Saved experiments to experiments/' + config.name)


def load_data(dataset_path, data_out, batch_size):
    face_detector = ort.InferenceSession('models/version-RFB-320.onnx')
    face_detector_input = face_detector.get_inputs()[0].name

    paths = pathlib.Path(dataset_path).glob('*.jpg')
    paths = sorted([x for x in paths])
    random.shuffle(paths)

    faces = []
    bmis = []

    if os.path.exists(data_out):
        shutil.rmtree(data_out)
    os.mkdir(data_out)

    current_batch = 0

    for image_path in paths:
        face = extract_face(image_path, face_detector, face_detector_input)
        if face is None:
            continue

        faces.append(face)

        bmi = extract_labels(image_path)
        bmis.append(bmi)

        if len(faces) == batch_size:
            write_batch(current_batch, faces, bmis, data_out)
            current_batch += 1
            faces.clear()
            bmis.clear()

    # Write any remaining data to the batch.
    if len(faces) > 0:
        write_batch(current_batch, faces, bmis, data_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='default_experiment', type=str)
    parser.add_argument('--train_dataset', default='tiny', type=str)
    parser.add_argument('--val_dataset', default='tiny', type=str)
    parser.add_argument('--data_out', default='out', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    main(parser.parse_args())
