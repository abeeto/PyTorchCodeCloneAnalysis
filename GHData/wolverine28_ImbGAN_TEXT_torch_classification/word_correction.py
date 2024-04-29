import re
import spacy
import nltk

contraction_dict = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is", "I'd": "I would", "I'd've": "I would have",
    "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will",  "i'll've": "i will have", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "this's": "this is", "that'd": "that would",
    "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are", "they've": "they have",
    "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

def clean_contractions(text, contraction_dict):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([contraction_dict[t] if t in contraction_dict else t for t in text.split(" ")])
    return text

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_dict = {
    "‘": "'",    "₹": "e",      "´": "'", "°": "",         "€": "e",
    "™": "tm",   "√": " sqrt ", "×": "x", "²": "2",        "—": "-",
    "–": "-",    "’": "'",      "_": "-", "`": "'",        '“': '"',
    '”': '"',    '“': '"',      "£": "e", '∞': 'infinity', 'θ': 'theta',
    '÷': '/',    'α': 'alpha',  '•': '.', 'à': 'a',        '−': '-',
    'β': 'beta', '∅': '',       '³': '3', 'π': 'pi'
}
def clean_special_chars(text, punct, punct_dict):
    for p in punct_dict:
        text = text.replace(p, punct_dict[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text



stopwords = nltk.corpus.stopwords.words('english')
def preprocess(df, contraction_dict, punct, punct_dict):
    texts = df['review']
    processed_texts = texts.apply(lambda x: x.lower())
    processed_texts = processed_texts.apply(lambda x: clean_contractions(x, contraction_dict))
    processed_texts = processed_texts.apply(lambda x: clean_special_chars(x, punct, punct_dict))
    processed_texts = processed_texts.apply(lambda x: re.split('\W+', x))
    processed_texts = processed_texts.apply(lambda x: [token for token in x if token not in stopwords])
    df['processed_review'] = processed_texts

def preprocess_text(text, contraction_dict, punct, punct_dict):
    clean_text=text.lower()
    clean_text=clean_contractions(clean_text, contraction_dict)
    clean_text=clean_special_chars(clean_text, punct, punct_dict)
    clean_text=re.split('\W+', clean_text)
    clean_text=[token for token in clean_text if token not in stopwords]
    return " ".join(clean_text)
# preprocess_text("samhdbei. 2345324@@# !~~~ sdne @ dsecwAADEk. SDKM",contraction_dict, punct, punct_dict)

nlp = spacy.load('en_core_web_sm', disable=['parser','tagger','ner'])

def tokenizer(s):
    return [w.text.lower() for w in nlp(preprocess_text(s,contraction_dict, punct, punct_dict))]