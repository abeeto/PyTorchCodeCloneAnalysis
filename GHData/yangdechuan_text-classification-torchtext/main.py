import time

import torch
from torch import optim
import torch.nn.functional as F
import torchtext
from sklearn import metrics

from modeling import KimCNN

MAX_LEN = 20
BATCH_SIZE = 64
NUM_EPOCHS = 5
NUM_CLASSES = 5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================== #
    #        Use torchtext to make data.          #
    # =========================================== #
    # Step 1. Define field
    TEXT = torchtext.data.Field(fix_length=MAX_LEN,
                                lower=True,
                                batch_first=True)
    LABEL = torchtext.data.Field(sequential=False,
                                 unk_token=None,  # Make sure index form 0.
                                 is_target=True)
    # Step 2: Load data.
    # Set fine_grained=True to use 5-class instead of 3-class labeling.
    train, val, test = torchtext.datasets.SST.splits(text_field=TEXT,
                                                     label_field=LABEL,
                                                     fine_grained=True)
    # Step 3: Build vocab.
    TEXT.build_vocab(train,
                     min_freq=1,
                     vectors=torchtext.vocab.GloVe(name="840B", dim=300))
    LABEL.build_vocab(train)
    # Step 4: Define iterator.
    train_iter, val_iter, test_iter = torchtext.data.Iterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
    )
    print("vocab size: {}".format(len(TEXT.vocab.itos)))

    # =========================================== #
    #           Build and train model             #
    # =========================================== #
    word_embedding = TEXT.vocab.vectors
    model = KimCNN(word_embedding=word_embedding,
                   num_classes=NUM_CLASSES)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)
    print("Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_iter.init_epoch()
        model.train()
        tic = time.time()
        for batch in train_iter:
            batch_xs = batch.text  # (batch_size, max_len)
            batch_ys = batch.label  # (batch_size, )
            batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)
            batch_out = model(batch_xs)  # (batch_size, num_classes)
            loss = F.cross_entropy(batch_out, batch_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time.time()

        # Validate every epoch.
        model.eval()
        y_true = []
        y_pred = []
        total_loss = 0
        batch_cnt = 0
        for batch in val_iter:
            batch_xs = batch.text  # (batch_size, max_len)
            batch_ys = batch.label  # (batch_size, )
            batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)
            batch_out = model(batch_xs)  # (batch_size, num_classes)
            batch_pred = batch_out.argmax(dim=-1)
            loss = F.cross_entropy(batch_out, batch_ys)
            total_loss += loss.item()

            y_true.extend(batch_ys.cpu().numpy())
            y_pred.extend(batch_pred.cpu().numpy())
            batch_cnt += 1

        accuracy = metrics.accuracy_score(y_true, y_pred)
        macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
        print("epoch {}, training use time {}s".format(epoch, toc - tic))
        print("val loss {:.5f}, val acc {:.3f}, val macro f1-score {:.3f}".format(total_loss / batch_cnt,
                                                                                  accuracy, macro_f1))

    print("Finish training!")
    # Test model.
    y_true = []
    y_pred = []
    for batch in test_iter:
        batch_xs = batch.text  # (batch_size, max_len)
        batch_ys = batch.label  # (batch_size, )
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)
        batch_out = model(batch_xs)  # (batch_size, num_classes)
        batch_pred = batch_out.argmax(dim=-1)
        loss = F.cross_entropy(batch_out, batch_ys)

        y_true.extend(batch_ys.cpu().numpy())
        y_pred.extend(batch_pred.cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    print("test accuracy {:.3f}, test macro f1-score {:.3f}".format(accuracy, macro_f1))
