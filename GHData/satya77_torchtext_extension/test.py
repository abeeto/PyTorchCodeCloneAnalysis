import torch
from Field import Field, NestedField
from Iterators import gen_bptt_iter
from LanagueModelDataset import LanguageModelingDataset, create_tokenizer, \
    text_cleaner, word_ids_to_sentence

USE_GPU = True
BATCH_SIZE = 5
folder = './wiki.train-sub.tokens'
cleaned_data = './wiki.train-sub-cleaned.tokens'
token_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_len = 15
clean_text = True
examples_to_consider=4

if __name__ == '__main__':
    if clean_text:
        text_cleaner(folder, cleaned_data)

    spacy_tok = create_tokenizer()  # create a custom tokenizer, can be replaced by your own tokenizer
    WORD = Field(pad_token = '<pad>', unk_token = "<unk>", tokenize = spacy_tok, lower = True,
                 sequential = True)  # sequential should be true

    CHAR_NESTING = Field(tokenize  = list, sequential = True, fix_length=token_len, unk_token = "<unk>", init_token = "<bow>",
                         eos_token = "<eow>")  # takes the word tokens and tokenize them into chars
    CHAR = NestedField(CHAR_NESTING, init_token = "<bos>", eos_token = "<eos>",
                       tokenize = spacy_tok)  # this one does the word tokens

    # put it into a dataset
    train_word = LanguageModelingDataset(cleaned_data, WORD)
    train_char = LanguageModelingDataset(cleaned_data, CHAR)

    # build a vocabulary
    WORD.build_vocab(train_word)
    CHAR.build_vocab(train_char)
    print("vocab is built! ")

    # make iterators for the batch, we need two iterators one for word and one for characters
    word_batcher, char_batcher = gen_bptt_iter((train_word, train_char), token_len = token_len, batch_size = BATCH_SIZE,
                                               bptt_len = sequence_len, device = device)

    count = 0
    print("Start batching...")
    for batch_word, batch_char in zip(word_batcher, char_batcher):
        count = count + 1
        print("word batch sizes:{}".format(batch_word.text.shape))
        words = word_ids_to_sentence(batch_word.text.type(torch.LongTensor), WORD.vocab, join=' ')
        print(words)
        print("char batch sizes:{}".format(batch_char.text.shape))
        chars = word_ids_to_sentence(batch_char.text.type(torch.LongTensor), CHAR.vocab, join=' ')
        print(chars)
        if count > examples_to_consider:
            break;
