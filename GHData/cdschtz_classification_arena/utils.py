import os
import torch


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ALL_CONFIGURATIONS = {
    "model_names": ('lstm', 'rnn', 'self_attention', 'cnn'),
    "dataset_names": ('WikiSyntheticGeneral', 'WikiSyntheticSophisticated', 'WikiNews'),
    "batch_sizes": (8, 16, 32, 64, 128),
    "learn_rates": (1e-3, 1e-2, 1e-4, 1e-5),
    "embedding_functions": ('glove', 'fasttext')
}


def get_files_with_data(path=os.getenv("WIKI_SYNTHETIC_PATH")):
    all_files = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if '.jsonl' in filename:
                all_files.append(os.path.join(root, filename))
            else:
                continue
    all_files = sorted(all_files)
    return all_files


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def plot_train_data(losses, accuracies):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(1, len(losses) + 1, 1)
    plt.plot(x, losses, color='r')
    # plt.show()
    plt.plot(x, accuracies / 100, color='b')
    plt.show()


def write_results(data):
    import datetime
    """Writes the file where all results are listed."""
    # general_info
    model_name, dataset_name = data['model_name'], data['dataset_name']
    # final scores
    train_acc, val_acc, test_acc = data['train_acc'], data['val_acc'], data['test_acc']
    # time information
    device = data['device']
    start, end, duration = data['time_info']['start'], data['time_info']['end'], data['time_info']['duration']
    # samples configuration
    splits, sizes = [0.7, 0.15, 0.15] if 'splits' not in data else data['splits'], data['sizes']
    # data process configuration
    tokenizer, embeddings = data['tokenizer'], data['embeddings']
    vocabulary, vocab_size = data['vocabulary'], data['vocab_size']
    # train configuration
    epochs, learn_rate = data['epochs'], data['learn_rate']
    batch_size, hidden_size = data['batch_size'], data['hidden_size']
    # graph data
    losses, accuracies = data['losses'], data['accuracies']

    from pathlib import Path
    base = f"./results/{model_name}"
    Path(base).mkdir(parents=True, exist_ok=True)
    path = f"{base}/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt"

    with open(path, 'w') as r_f:
        import json
        r_f.write(f"Model Type: {model_name.upper()}\nDataset: {dataset_name}\n\n\n")
        r_f.write("SCORES\n{:20}|{:20}|{:20}\n{:20}|{:20}|{:20}\n\n\n".
                  format("Train Acc", "Val Acc", "Test Acc", str(train_acc)[:5], str(val_acc)[:5], str(test_acc)[:5]))
        r_f.write("TIME\n{:20}|{:20}|{:20}|{:20}\n{:20}|{:20}|{:20}|{:20}\n\n\n".
                  format("Device", "Duration", "Start", "End", device, duration, start, end))
        r_f.write("SAMPLES\n{:20}|{:20}|{:20}|{:20}\n{:20}|{:20}|{:20}|{:20}\n\n\n".
                  format("Total", "Training", "Validation", "Test",
                         str(sizes[0]), str(sizes[1]), str(sizes[2]), str(sizes[3])))
        r_f.write("DATA PROCESS CONFIGURATION\n{:20}|{:20}|{:40}\n{:20}|{:20}|{:40}\n\n\n".
                  format("Tokenizer", "Embeddings", "Vocabulary", tokenizer, embeddings,
                         vocabulary + f" (size: {vocab_size})"))
        r_f.write("TRAIN CONFIGURATION\n{:20}|{:20}|{:20}|{:20}\n{:20}|{:20}|{:20}|{:20}\n\n\n".
                  format("Epochs", "Learn Rate", "Batch Size", "Hidden Size",
                         str(epochs), str(learn_rate), str(batch_size), str(hidden_size)))
        r_f.write("GRAPH DATA\n{}".format(json.dumps({"loss": losses, "acc": accuracies})))
