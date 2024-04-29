import os
from collections import defaultdict, Counter


here = lambda *x: os.path.abspath(os.path.join(os.path.dirname(__file__), *x))
corpus_dir = here('corpus', )
lengths_file = here('lengths', )
index_file = here('index',)


corpus_length = 0
frequencies = defaultdict(list)
with open(lengths_file, "w+") as documents_lengths:
    with open(index_file, "w+") as index:
        for file in os.listdir(corpus_dir):
            current_file = os.path.join(corpus_dir, file)
            with open(current_file, 'r') as f:
                terms = map(lambda line: line.split(" "), f.readlines())
                terms = reduce(lambda x, y: x.extend(y), terms)
                length = len(terms)
                corpus_length += length
                documents_lengths.write('%d\n' % length)
                counter = Counter(terms)
                for term, frequency in counter.items():
                    frequencies[term].append((file, frequency))
        frequencies = sorted(frequencies.iteritems(), key=lambda x: len(x[1]))
        for entry in frequencies:
            index.write('{0} {1} '.format(entry[0], len(entry[1])))
            for document in entry[1]:
                index.write('{0} {1} '.format(document[0], document[1]))
            index.write('\n')
    documents_lengths.write('%d\n' % corpus_length)
