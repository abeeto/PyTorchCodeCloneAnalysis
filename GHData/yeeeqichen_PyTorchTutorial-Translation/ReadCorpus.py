from Config import config
import re
import unicodedata

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


fra_sentences = []
eng_sentences = []
eng_dict = dict()
eng_dict['#'] = 0
fra_dict = dict()
fra_dict['#'] = 0
with open(config.data_path) as f:
    for line in f:
        sent1, sent2 = line.strip('\n').split('\t')
        eng = normalize_string(sent1).split(' ')
        fra = normalize_string(sent2).split(' ')
        if len(eng) <= 10 and len(fra) < 10:
            eng_sentences.append(eng)
            fra_sentences.append(fra)


# build dictionary
idx = 1
for sent in eng_sentences:
    for word in sent:
        if word not in eng_dict:
            eng_dict[word] = idx
            idx += 1
eng_dict['[PAD]'] = idx
config.enc_vocab_size = idx + 1
idx = 1
for sent in fra_sentences:
    for word in sent:
        if word not in fra_dict:
            fra_dict[word] = idx
            idx += 1
fra_dict['[PAD]'] = idx
config.dec_vocab_size = idx + 1
