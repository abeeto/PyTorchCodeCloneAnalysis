import re
from Lang import Lang


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"(\[.+?\])", r" ", s)  # Removing square brackets and its content (not recursively)
    s = re.sub(r"(\(.+?\))", r" ", s)  # Removing brackets and its content (not recursively)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Zàèéìòù.!?]+", r" ", s)
    return s


def filter_pair(p, MAX_LENGTH):
    # Filtering out too long sentences
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs, MAX_LENGTH):
    # Filtering out pairs with too long sentences
    return [pair for pair in pairs if filter_pair(pair, MAX_LENGTH)]


def read_langs(lang1, lang2, dataset, mode):
    # Reading the file and splitting into lines
    print("Reading lines...")
    lines = open('dataset/%s_%s.en-it.txt' % (dataset, mode)).read().strip().split('\n')

    # Splitting every line into sentences pair and normalizing
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Returning sentences pairs (and possibly language structures)
    if lang1 is None and lang2 is None:
        input_lang = Lang('en')
        output_lang = Lang('it')
        return input_lang, output_lang, pairs
    else:
        return pairs


def prepare_data(lang1, lang2, dataset, mode, MAX_LENGTH):
    # Generating language structures and sentence pairs
    if lang1 is None and lang2 is None:
        lang1, lang2, pairs = read_langs(lang1, lang2, dataset, mode)
    else:
        pairs = read_langs(lang1, lang2, dataset, mode)
    print("Read %s sentence pairs" % len(pairs))

    # Filtering out too long sentences
    pairs = filter_pairs(pairs, MAX_LENGTH)
    print("Trimmed to %s sentence pairs" % len(pairs))

    # Indexing words into language structures
    if mode != 'test':
        print("Indexing words...")
        for pair in pairs:
            lang1.index_words(pair[0])
            lang2.index_words(pair[1])

    return lang1, lang2, pairs
