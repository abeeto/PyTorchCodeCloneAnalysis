import torch
from utilities.generate_response import GenerateResponse
from models.seq2seq_transformer import Seq2Seq
from utilities.vocab import TanakaVocabs


TOP_WORDS = 80000
CHAR2ID_FILEPATH = "utilities/char2id.model"
MODEL_FILEPATH = "output/posmodel_epoch30.pth"
MAXLEN = 60


def main():
    vocabs = TanakaVocabs(top_words=TOP_WORDS)
    vocabs.load_char2id_pkl(CHAR2ID_FILEPATH)

    input_dim = len(vocabs.vocab_X.char2id)
    output_dim = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=MAXLEN + 8)
    model.load_state_dict(torch.load(MODEL_FILEPATH))

    gr = GenerateResponse(model, vocabs)
    text = "input"

    while text != "":
        text = input("message : ")
        if text == "":
            break

        print("response : {}".format(gr(text)))
        print("-" * 10)


if __name__ == "__main__":
    main()
