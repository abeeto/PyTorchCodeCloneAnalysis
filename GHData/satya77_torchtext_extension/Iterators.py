
from torchtext import data
import torchtext
import math


class BPTTIterator_CHAR(data.BPTTIterator):
    def __init__(self,token_len,dataset, batch_size, bptt_len, **kwargs):
        self.token_len=token_len
        self.bptt_len= bptt_len
        super(data.BPTTIterator, self).__init__(dataset, batch_size,bptt_len, **kwargs)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None

        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size) * self.batch_size - len(text)))

        data = TEXT.numericalize(text, device=self.device)
        dataset = torchtext.data.Dataset(examples=self.dataset.examples, fields=[('text', TEXT), ('target', TEXT)])

        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = self.bptt_len
                if i + 1 + seq_len * self.batch_size <= len(data):
                    batch_text = data[i:i + seq_len * self.batch_size].view(self.batch_size, self.token_len, -1).contiguous()
                    batch_target = data[i + 1:i + 1 + seq_len * self.batch_size].view(self.batch_size, self.token_len,
                                                                                      -1).contiguous()

                yield torchtext.data.Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return


class BPTTIterator_WORD(data.BPTTIterator):
    """custom iterator for hybrid char-word"""

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize([text], device=self.device)
        dataset = torchtext.data.Dataset(examples=self.dataset.examples, fields=[('text', TEXT), ('target', TEXT)])
        TEXT.batch_first = True
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = self.bptt_len
                if i + 1 + seq_len * self.batch_size <= len(data):
                    batch_text = data[i:i + seq_len * self.batch_size].view(self.batch_size, -1).contiguous()
                    batch_target = data[i + 1:i + 1 + seq_len * self.batch_size].view(self.batch_size,
                                                                                      -1).contiguous()
                yield torchtext.data.Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return


def gen_bptt_iter(dataset,token_len, batch_size, bptt_len, device):
    print("Generating Word Batcher....")
    word_batcher=BPTTIterator_WORD(dataset[0], batch_size, bptt_len, device=device)
    print("Done")
    print("Generating Char Batcher....")
    char_batcher=BPTTIterator_CHAR(token_len,dataset[1], batch_size, bptt_len,device=device)
    print("Done")
    return word_batcher,char_batcher
    # for batch_word, batch_char in zip(word_batcher,char_batcher):
    #     yield batch_word.text, batch_char.text, batch_word.target, batch_char.target

