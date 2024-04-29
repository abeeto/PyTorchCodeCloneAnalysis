from allennlp.modules.elmo import Elmo, batch_to_ids
import func
import torch
import utils
from diskdict import DiskDict

class ElmoEmbedding:
    def __init__(self, dim):
        if dim == 2048:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        elif dim == 512:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        self.dim = dim
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        if func.gpu_available():
            self.elmo = self.elmo.cuda()
        self.elmo.eval()
        self.load()


    def save(self):
        pass


    def load(self):
        self.cache = DiskDict(f'./generate/elmo.{self.dim}.cache')


    def convert(self, sentences):
        not_hit = set()
        for sent in sentences:
            key = self.make_key(sent)
            if key not in self.cache:
                not_hit.add(key)
        not_hit = list(not_hit)
        if not_hit:
            embeddings, masks = self.convert_impl([self.make_sentence(key) for key in not_hit])
            for key, embedding, mask in zip(not_hit, torch.unbind(embeddings), torch.unbind(masks)):
                embedding = embedding[:mask.sum()]
                self.cache[key] = embedding.tolist()
        embeddings = [func.tensor(self.cache[self.make_key(sent)]) for sent in sentences]
        mlen = max([e.shape[0] for e in embeddings])
        embeddings = [func.pad_zeros(e, mlen, 0) for e in embeddings]
        embeddings = torch.stack(embeddings)
        assert embeddings.requires_grad == False
        return embeddings


    def make_key(self, sent):
        return '$$'.join(sent)


    def make_sentence(self, key):
        return key.split('$$')


    def convert_impl(self, sentences):
        character_ids = func.tensor(batch_to_ids(sentences))
        m = self.elmo(character_ids)
        embeddings = m['elmo_representations']
        embeddings = torch.cat(embeddings, -1)
        mask = m['mask']
        return embeddings, mask


if __name__ == '__main__':
    import torch

    elmo = ElmoEmbedding(512)
    # use batch_to_ids to convert sentences to character ids
    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    embeddings = elmo.convert(sentences)
    print(embeddings.shape)
    # m['elmo_representations'] is length two list of tensors.
    # Each element contains one layer of ELMo representations with shape
    # (2, 3, 1024).
    #   2    - the batch size
    #   3    - the sequence length of the batch
    #   1024 - the length of each ELMo vector