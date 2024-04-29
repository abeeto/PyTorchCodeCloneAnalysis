from __future__ import division
import os
import sys
import string
import time
from heapq import heappush, heappop
from collections import defaultdict
from stemmer import PorterStemmer
from pairwise import pairwise


here = lambda *x: os.path.abspath(os.path.join(os.path.dirname(__file__), *x))
stopwords_file = here('stopwords', )
index_file = here('index',)
lengths_file = here('lengths', )

top = 20
lambda_ = 0.5
query = sys.argv[1:]
inner_query = []
p = PorterStemmer()
index = {}
count_term = {}


start_time = time.time()
with open(stopwords_file, "r") as file:
    stopwords = map(lambda line: line.strip(), file.readlines())

with open(index_file, "r") as index_file:
    lines = index_file.readlines()
    for line in lines:
        entry = line.split(" ")
        documents = entry[2:]
        dictionary = defaultdict(int)
        for document, count in pairwise(documents):
            dictionary[document] = int(count)
        count_term.update({entry[0]: int(entry[1])})
        index.update({entry[0]: dictionary})

with open(lengths_file, "r") as lengths:
    documents_lengths = map(lambda line: line.strip(), lengths.readlines())
    documents_lengths = [int(length) for length in documents_lengths]
    corpus_length = documents_lengths[len(documents_lengths) - 1]
    del documents_lengths[len(documents_lengths) - 1]


for term in query:
    term = term.translate(None, string.punctuation)
    if term not in stopwords:
        inner_query.append(p.stem(term.lower(), 0, len(term) - 1))


documents_count = len(documents_lengths)
query_terms_count = len(inner_query)
heap = []
probabilities = [[1 for x in xrange(documents_count)] for x in xrange(query_terms_count)]
results = [1 for x in xrange(documents_count)]
for i, term in enumerate(inner_query):
    try:
        term_collection_frequency = count_term[term] / corpus_length
    except KeyError:  # the term doesn't appear in the collection
        continue
    for document in xrange(1, documents_count + 1):
        probabilities[i][document - 1] = lambda_ * (index[term][str(document)] / documents_lengths[document - 1] + term_collection_frequency)


for i in xrange(documents_count):
    for j in xrange(query_terms_count):
        results[i] *= probabilities[j][i]
    heappush(heap, (-results[i], i + 1))


print 'Top {0} results ({1} seconds)'.format(top, time.time() - start_time)
print 'Results matching query {}'.format(' '.join(inner_query))
while(top):
    top -= 1
    value, document = heappop(heap)
    print 'Document {0} with probability {1}'.format(document, -value)
