import re
from janome.tokenizer import Tokenizer
import torchtext

j_t = Tokenizer()


def tokenizer_janome(text):
    return [tok for tok in j_t.tokenize(text, wakati=True)]


def preprocessing_text(text):
    # 半角・全角の統一

    # 英語の小文字化
    # output = output.lower()

    # 改行・半角スペース・全角スペースの削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub(' ', '', text)
    text = re.sub('　', '', text)

    # 数字文字を0に
    text = re.sub(r'[0-9 ０-９]', '0', text)

    # 記号と数字の除去

    # 特定文字を正規表現で置換

    return text


def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_janome(text)
    return ret


max_length = 25
TEXT = torchtext.data.Field(sequential=True,                        # データの長さが可変か
                            tokenize=tokenizer_with_preprocessing,  # 文書を読み込んだ際にtokenizeする関数を指定
                            use_vocab=True,                         # 単語をボキャブラリーに追加するかどうか
                            lower=True,                             # アルファベットを小文字に変換するかどうか
                            include_lengths=True,                   # 文章の単語数のデータを保持するか
                            batch_first=True,                       # ミニバッチの次元を先頭に用意するか
                            fix_length=max_length)                  # 全部の文章を指定された長さにするか
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='text_train.tsv', validation='text_val.tsv',
    test='text_test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

# ボキャブラリーを作成
TEXT.build_vocab(train_ds, min_freq=1)

# DataLoaderの作成
train_dl = torchtext.data.Iterator(train_ds, batch_size=2, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=2, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=2, train=False, sort=False)
