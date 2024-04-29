import os


DATA_SIZE = 1
TRAIN_SIZE = 0.9
VAL_SIZE = 0.7
TOP_WORDS = 80000

MAXLEN = 60
BATCH_SIZE = 100
EPOCH_SIZE = 100

dataset_train_pkl = "dataloader/twitter_dataset_normal_train.model"
dataset_val_pkl = "dataloader/twitter_dataset_normal_val.model"
dataset_test_pkl = "dataloader/twitter_dataset_normal_test.model"


def get_dataset():
    from sklearn.model_selection import train_test_split
    from dataloader.twitter_dataset import TwitterDataset
    from dataloader.twitter_transform import TwitterTransform
    from utilities.functions import load_json

    nml = load_json("output/normal.json")
    dataset_X = []
    dataset_y = []
    for dialogue in nml:
        msg = dialogue[0]
        res = dialogue[1]

        if len(msg.split()) > MAXLEN:
            continue
        if len(res.split()) > MAXLEN:
            continue

        dataset_X.append(msg)
        dataset_y.append(res)

    X_train, X_other, y_train, y_other = train_test_split(
        dataset_X, dataset_y, test_size=(1 - TRAIN_SIZE)
    )
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=(1 - VAL_SIZE))

    train_dataset = TwitterDataset(MAXLEN, TwitterTransform())
    train_dataset.messages = X_train
    train_dataset.responses = y_train

    val_dataset = TwitterDataset(MAXLEN, TwitterTransform())
    val_dataset.messages = X_val
    val_dataset.responses = y_val

    test_dataset = TwitterDataset(MAXLEN, TwitterTransform())
    test_dataset.messages = X_test
    test_dataset.responses = y_test

    return train_dataset, val_dataset, test_dataset


def transform_dataset_and_get_vocab(all_datasets):
    from utilities.vocab import TanakaVocabs
    from utilities.constant import CHAR2ID

    vocabs = TanakaVocabs(TOP_WORDS)
    vocabs.load_char2id_pkl(CHAR2ID)

    if (
        os.path.exists(dataset_train_pkl)
        and os.path.exists(dataset_val_pkl)
        and os.path.exists(dataset_test_pkl)
    ):
        pass
    else:
        vocabs.transform(all_datasets)
        all_datasets[0].save_corpus_pkl(dataset_train_pkl)
        all_datasets[1].save_corpus_pkl(dataset_val_pkl)
        all_datasets[2].save_corpus_pkl(dataset_test_pkl)

    return vocabs, all_datasets


def main():
    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Datasetの作成
    # --------------------------------------
    train_dataset, val_dataset, test_dataset = get_dataset()
    all_datasets = [train_dataset, val_dataset, test_dataset]

    # --------------------------------------
    # Vocabの作成
    # --------------------------------------
    vocabs, all_datasets = transform_dataset_and_get_vocab(all_datasets)
    train_dataset, val_dataset, test_dataset = all_datasets

    # --------------------------------------
    # DataLoaderの作成
    # --------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = TanakaDataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(test_dataset, batch_size=1, random_state=0)

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transformer import Seq2Seq

    input_dim = len(vocabs.vocab_X.char2id)
    output_dim = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=60 + 8, output_filename="normal")

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import torch
    import pytorch_lightning as pl

    # from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from multiprocessing import freeze_support
    from utilities.callbacks import DisplayPredictDialogue
    from utilities.constant import SAVE_MODELS_PTH

    freeze_support()
    trainer = pl.Trainer(
        callbacks=[
            DisplayPredictDialogue(
                vocabs,
                TanakaDataLoader(train_dataset, batch_size=1, random_state=0),
                TanakaDataLoader(test_dataset, batch_size=1, random_state=0),
            )
        ],
        # strategy=DDPStrategy(find_unused_parameters=False)
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        logger=TensorBoardLogger(
            os.getcwd(), version="B{}_E{}_S{}_normal".format(BATCH_SIZE, EPOCH_SIZE, DATA_SIZE)
        ),
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

    torch.save(model.state_dict(), SAVE_MODELS_PTH)


if __name__ == "__main__":
    main()
