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
#
# -----------------------------------
# Data Info
# -----------------------------------
# case : neg
#     train_dialogue : 25497 turns
#     val_dialogue : 7649 turns
#     test_dialogue : 3279 turns
# case : pos
#     train_dialogue : 12694 turns
#     val_dialogue : 3808 turns
#     test_dialogue : 1633 turns
#
import os
import random
from typing import List, Tuple
from dataloader.twitter_dataset import TwitterDataset


DATA_SIZE = 1
SPLIT_RATIO = 0.2
VAL_SIZE = 0.7
TOP_WORDS = 80000

BATCH_SIZE = 50
EPOCH_SIZE = 100
MAXLEN = 60

NEGPOS = "pos"
USE_CORPUS = "output/{}.json".format(NEGPOS)
NEG_CORPUS = "output/neg.json"
POS_CORPUS = "output/pos.json"
NML_CORPUS = "output/normal.json"
BASE_STATE_DIC = "output/normalmodel_epoch30.pth"


def split_train_val_test(filepath, train_size=0.7, val_size=0.7):
    import random
    from utilities.functions import load_json

    train_dialogue = load_json(filepath)
    neg_dialogue = load_json(NEG_CORPUS)
    pos_dialogue = load_json(POS_CORPUS)
    nml_dialogue = load_json(NML_CORPUS)

    other_dialogue = []
    other_dialogue.extend(neg_dialogue)
    other_dialogue.extend(pos_dialogue)
    other_dialogue.extend(nml_dialogue)
    random.shuffle(other_dialogue)

    train_size = len(train_dialogue)
    other_dialogue = other_dialogue[: int(train_size * SPLIT_RATIO)]

    val_dialogue = other_dialogue[: int(len(other_dialogue) * VAL_SIZE)]
    test_dialogue = other_dialogue[int(len(other_dialogue) * VAL_SIZE) : int(len(other_dialogue))]

    val_size = len(val_dialogue)
    test_size = len(test_dialogue)

    print("train_size : {}".format(train_size))
    print("val_size : {}".format(val_size))
    print("test_size : {}".format(test_size))

    return train_dialogue, val_dialogue, test_dialogue


def get_datasets(all_dialogues: List[list]) -> List[TwitterDataset]:
    def get_dataset_using_dialogue(dialogue: List[Tuple]):

        dataset = TwitterDataset()
        dataset.messages = [d[0] for d in dialogue]
        dataset.responses = [d[1] for d in dialogue]
        return dataset

    train_dialogue = all_dialogues[0]
    val_dialogue = all_dialogues[1]
    test_dialogue = all_dialogues[2]

    train_dataset = get_dataset_using_dialogue(train_dialogue)
    val_dataset = get_dataset_using_dialogue(val_dialogue)
    test_dataset = get_dataset_using_dialogue(test_dialogue)

    return train_dataset, val_dataset, test_dataset


def get_vocab_and_transform(all_datasets: List[TwitterDataset]):
    from utilities.vocab import TanakaVocabs
    from utilities.constant import CHAR2ID

    vocabs = TanakaVocabs(TOP_WORDS)
    vocabs.load_char2id_pkl(CHAR2ID)

    for dataset in all_datasets:
        dataset.messages = vocabs.X_transform(dataset.messages)
        dataset.responses = vocabs.y_transform(dataset.responses)

    return all_datasets, vocabs


def main():
    # -------------------------------------------------
    # CORPUS_FILEからtrain, val, test文のターンを分割する
    # train : List(message, response)
    # -------------------------------------------------
    train_dialogue, val_dialogue, test_dialogue = split_train_val_test(USE_CORPUS)

    # -------------------------------------------------
    # 分割したターンリストからTwitterDatasetを作成する
    # -------------------------------------------------
    all_dialogues = [train_dialogue, val_dialogue, test_dialogue]
    train_dataset, val_dataset, test_dataset = get_datasets(all_dialogues)

    # -------------------------------------------------
    # Vocabを使って単語をベクトル化する(one-hot)
    # -------------------------------------------------
    all_datasets = [train_dataset, val_dataset, test_dataset]
    all_datasets, vocabs = get_vocab_and_transform(all_datasets)

    # -------------------------------------------------
    # DataLoaderの作成
    # -------------------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(all_datasets[0], batch_size=BATCH_SIZE)
    val_dataloader = TanakaDataLoader(all_datasets[1], batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(all_datasets[2], batch_size=1, random_state=0)

    # -------------------------------------------------
    # モデルの作成
    # -------------------------------------------------
    from models.seq2seq_transformer import Seq2Seq

    src_vocab_size = len(vocabs.vocab_X.char2id)
    tgt_vocab_size = len(vocabs.vocab_y.char2id)
    model = Seq2Seq(src_vocab_size, tgt_vocab_size, maxlen=60 + 8, output_filename=NEGPOS)

    # -------------------------------------------------
    # トレーニング
    # -------------------------------------------------
    import torch
    import pytorch_lightning as pl

    # from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from multiprocessing import freeze_support
    from utilities.callbacks import DisplayPredictDialogue
    from utilities.constant import SAVE_MODELS_PTH

    model.load_state_dict(torch.load(BASE_STATE_DIC))

    freeze_support()
    trainer = pl.Trainer(
        callbacks=[
            DisplayPredictDialogue(
                vocabs,
                TanakaDataLoader(train_dataset, batch_size=1, random_state=0),
                TanakaDataLoader(test_dataset, batch_size=1, random_state=0),
                filename=NEGPOS,
            )
        ],
        # strategy=DDPStrategy(find_unused_parameters=False)
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        logger=TensorBoardLogger(
            os.getcwd(),
            version="B{}_E{}_S{}_{}".format(BATCH_SIZE, EPOCH_SIZE, DATA_SIZE, NEGPOS),
        ),
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

    torch.save(model.state_dict(), "output/pos_model.pth")


if __name__ == "__main__":
    main()
