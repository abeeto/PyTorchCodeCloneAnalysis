from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

ps = PorterStemmer()

def change(s, ps=ps):
    if not s.endswith('ing'):
        s = ps.stem(s)
    return s



def preprocess_text(s):
    s = s.lower()
    for ch in string.punctuation:
        s = s.replace(ch, '')

    s = s.split() + ['<end>']
    s = [x for x in s if x != '']

    return s
