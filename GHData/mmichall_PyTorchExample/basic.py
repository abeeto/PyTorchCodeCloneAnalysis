# import nltk
from nltk.corpus import brown
import gensim

# nltk.download('brown')

w2v = gensim.models.KeyedVectors.load_word2vec_format('F:\GoogleNews-vectors-negative300.bin', binary=True)

news_words = [word for word in brown.words(categories='mystery') if word in w2v.wv]
news_embedding = [w2v.wv[word] for word in news_words]

print(news_embedding)

