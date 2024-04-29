from __future__ import division
from __future__ import print_function
import time

import torch
import torch.nn as nn

from utils.utils import *
from gcn import GCN

from config import CONFIG

cfg = CONFIG()

if len(sys.argv) != 2:
    sys.exit("No datasets given!\n\tUse: python fit_model.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("Wrong dataset name")
cfg.dataset = dataset

# Set random seed
seed = 970918
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, tesmask, train_size, test_size = load_corpus(cfg.dataset)

features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
features = preprocess_features(features)
A_tilde = preprocess_adj(adj)


def convert_coo_to_tensor(mat):
    indices = torch.LongTensor(np.vstack((mat.row, mat.col)))
    v = torch.FloatTensor(mat.data)
    return torch.sparse.FloatTensor(indices, v, torch.Size(features.shape))


def repeat_mask_vertically(mask, label_count):
    """
    This function gets a zero-one numpy array mask
    flips it and repeats it for 'label_count' times

    :returns:
    mask flipped, mask flipped and repeated
    """
    once = torch.from_numpy(mask.astype(np.float32))
    repeated = torch.transpose(torch.unsqueeze(once, 0), 1, 0).repeat(1, label_count)
    return once, repeated


features = convert_coo_to_tensor(features)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)
y_test = torch.from_numpy(y_test)
train_mask, m_train_mask = \
    repeat_mask_vertically(train_mask.astype(np.float32), y_train.shape[1])
A_tilde = convert_coo_to_tensor(A_tilde)

model = GCN(input_dim=features.shape[0], A_tilde=A_tilde, num_classes=y_train.shape[1], dropout_rate=0.5)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def evaluate(features, labels, mask):
    """
    :param features:
    A tensor containing the input features of the GCN

    :param mask:
    The vertex nodes or documents we wish to evaluate. It is represented as
    a '0' and '1' matrix with ones on the indices where we want to evaluate.

    :param labels:
    The actual labels of each document node. The labels are represented in one-hot
    meaning e_i in node 'v' shows document 'v' is in class i

    :returns:
    [loss], [accuracy], [predicted labels], [actual labels], [duration of evaluation]
    """
    global loss_func
    evaluation_start = time.time()

    # Evaluation mode:
    model.eval()
    with torch.no_grad():
        model_out = model(features)
        predicted = torch.max(model_out, 1)[1]
        mask, m_mask = repeat_mask_vertically(mask.astype(np.float32), labels.shape[1])
        loss = loss_func(model_out * m_mask, torch.max(labels, 1)[1])
        accuracy = ((predicted == torch.max(labels, 1)[1]).float() * mask).sum().item() / mask.sum().item()

    return loss.numpy(), accuracy, predicted.numpy(), labels.numpy(), (time.time() - evaluation_start)


validation_losses = []

# Train model
for epoch in range(cfg.epochs):

    """
    Epochs:
    The epochs may stop before 200 iterations if the loss of the validation data becomes worst
    than the last 10 validation data losses.
    """

    epoch_begin_time = time.time()

    # Forward pass
    model_out = model(features)
    model_prediction = torch.max(model_out, 1)[1]
    ground_truth = torch.max(y_train, 1)[1]
    loss = loss_func(model_out * m_train_mask, torch.max(y_train, 1)[1])
    acc = ((model_prediction == ground_truth).float() * train_mask).sum().item() / train_mask.sum().item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(features, y_val, val_mask)
    validation_losses.append(val_loss)

    print("\tEpoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}" \
              .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - epoch_begin_time))

    # We stop early if the validation loss becomes more than the mean of the last "early_stopping" validation
    # losses, This reduces the chance to overfit!
    if epoch > cfg.early_stopping and validation_losses[-1] > np.mean(validation_losses[-(cfg.early_stopping + 1):-1]):
        print("-Early stopping to avoid overfitting...")
        break

print("[Done] Optimization Finished!")

# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(features, y_test, tesmask)
print("\tTest set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))


def get_word_embeddings(model):
    global adj, train_size, test_size
    hidden_layer = model.layer1.embedding.numpy()
    return hidden_layer[train_size:adj.shape[0] - test_size]


def get_doc_embeddings(model):
    global adj, train_size, test_size
    hidden_layer = model.layer1.embedding.numpy()
    return np.vstack((hidden_layer[:train_size], hidden_layer[adj.shape[0] - test_size:]))


# save document embeddings
doc_embeddings = get_doc_embeddings(model)
doc_vectors = []
doc_id = 0
for i in range(len(doc_embeddings)):
    doc_vector = doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)
