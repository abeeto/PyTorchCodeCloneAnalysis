# --------------------------------------
# 定数
# ---------------------------------------
# DATA_SIZE : 使用するコーパスファイルの割合
# TRAIN_SIZE : コーパスファイルのうち，トレーニングで利用する割合
# VAL_SIZE : トレーニングで利用しないコーパスファイルのうち，検証で利用する割合．
#            残り，つまり1-VAL_SIZEは，テストデータで利用する．
# TOP_WORDS : 出現頻度上位TOP_WORDSの語彙を使って学習．他はUNKトークンとして配置．
# BATCH_SIZE : バッチサイズ
# EPOCH_SIZE : 最大エポックサイズ
DATA_SIZE = 1
TRAIN_SIZE = 0.9
VAL_SIZE = 0.7
TOP_WORDS = 80000

MAXLEN = 60
BATCH_SIZE = 100
EPOCH_SIZE = 100


import os
import random


# グローバル変数
dataset_train_pkl = "dataloader/twitter_dataset_S{}_train.model".format(DATA_SIZE)
dataset_val_pkl = "dataloader/twitter_dataset_S{}_val.model".format(DATA_SIZE)
dataset_test_pkl = "dataloader/twitter_dataset_S{}_test.model".format(DATA_SIZE)


def get_dataset():
    from dataloader.twitter_transform import TwitterTransform
    from dataloader.twitter_dataset import TwitterDataset
    from utilities.functions import train_val_test

    train_files, val_files, test_files = train_val_test(
        all_size=DATA_SIZE, train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )

    transform = TwitterTransform()
    train_dataset = TwitterDataset(limit_len=MAXLEN, transform=transform)
    val_dataset = TwitterDataset(limit_len=MAXLEN, transform=transform)
    test_dataset = TwitterDataset(limit_len=MAXLEN, transform=transform)

    # train_datasetのロード
    if os.path.exists(dataset_train_pkl):
        train_dataset.load_corpus_pkl(dataset_train_pkl)
        print("train_dataset content")
        for _ in range(5):
            i = random.randint(0, len(train_dataset))
            print(train_dataset.messages[i])
            print(train_dataset.responses[i])
    else:
        train_dataset.load_corpus(train_files)

    # val_datasetのロード
    if os.path.exists(dataset_val_pkl):
        val_dataset.load_corpus_pkl(dataset_val_pkl)
        print("val_dataset content")
        for _ in range(5):
            i = random.randint(0, len(val_dataset))
            print(train_dataset.messages[i])
            print(train_dataset.responses[i])
    else:
        val_dataset.load_corpus(val_files)

    # test_datasetのロード
    if os.path.exists(dataset_test_pkl):
        test_dataset.load_corpus_pkl(dataset_test_pkl)
        print("test_dataset content")
        for _ in range(5):
            i = random.randint(0, len(test_dataset))
            print(test_dataset.messages[i])
            print(test_dataset.responses[i])
    else:
        test_dataset.load_corpus(test_files)

    return train_dataset, val_dataset, test_dataset


def transform_dataset_and_get_vocab(all_datasets):
    from utilities.vocab import TanakaVocabs
    from utilities.constant import CHAR2ID

    vocabs = TanakaVocabs(TOP_WORDS)

    if os.path.exists(dataset_train_pkl) and os.path.exists(CHAR2ID):
        vocabs.load_char2id_pkl(CHAR2ID)
    else:
        vocabs.fit_transform(all_datasets, True, True, True, True)

        all_datasets[0].save_corpus_pkl(dataset_train_pkl)
        all_datasets[1].save_corpus_pkl(dataset_val_pkl)
        all_datasets[2].save_corpus_pkl(dataset_test_pkl)

        vocabs.save_char2id_pkl(CHAR2ID)

    return vocabs


def main():
    import os

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
    vocabs = transform_dataset_and_get_vocab(all_datasets)

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

    model = Seq2Seq(input_dim, output_dim, maxlen=60 + 8)

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
            os.getcwd(), version="B{}_E{}_S{}".format(BATCH_SIZE, EPOCH_SIZE, DATA_SIZE)
        ),
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model, test_dataloader)

    torch.save(model.state_dict(), SAVE_MODELS_PTH)


if __name__ == "__main__":
    main()
