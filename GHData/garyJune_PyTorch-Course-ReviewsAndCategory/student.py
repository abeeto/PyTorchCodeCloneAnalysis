#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

################################################################################
############################# Response to Question #############################
################################################################################
"""
Group Name: Group 1
Members: 
    - Xinli Wang
    - Kan-Lin Lu (z3417618) 

Question:
Briefly describe how your program works, and explain any design and training decisions you made along the way

Response:
Firstly, our program performed the tokenization and preprocessing steps of the input samples. The idea of preprocessing 
is to remove any special characters, URLs and English stop words to reduce the unwanted noises in our data samples. 
Normalization of the word samples by converting all samples into lower-case words was also applied in attempt to increase
the overall performance of the model. Secondly, we constructed the class network which uses the Bidirectional LSTM 
(bi-lstm) architecture. We chose the Bi-LSTM neural network because of its ability to preserve long-term information
and addresses the vanishing gradient problem encountered by many traditional RNNs. The bi-lstm neural network compose 
of LSTM units that operate in both directions to incorporate past and future context information without retaining 
duplicated context information. We used two different fully connected linear layer where one layer predicts the rating 
probability, and the other layer predicts the 5 probabilities associated with the 5 respective business categories. 

In the forward function of the class network we padded each sequences in the batch so that they all have the same length 
and then this padded sequence is passed through the bi-lstm model. We then concatenated the forward and reversed sequences 
to incorporate the past and future context information without retaining any duplicated information. Dropout with 
probability of 0.5 was applied on the concatenated sequences before been feed into the two respective fully connected 
linear layers. Sigmoid and log SoftMax activation functions were used on the predicted rating and category output, 
respectively. In the class loss, we used binary_cross_entropy for calculating the rating loss given binary nature of 
the prediction and used cross_entropy for calculating the loss for the category prediction. The choice of the Sigmoid 
activation coupled with binary_cross_entropy loss function and log SoftMax with cross_entropy loss functions were chosen 
due to the nature of the binary and multiclass classification problem we were trying to solve. The sum of these two 
losses were returned by the class loss. Finally, the class network outputs the predicted probabilities for the ratings 
and categories, respectively.

Most of the decisions are made by trial and error, mainly comparing the weighted score, note slight flutation does occur
under the same setting. We start with choosing network parametrs, then training parameters:

Network Parameters:

In order to examined the choice of dimensions for vector, under consistent setting (Note we select 5 epoch, batch size of 
64 with validation ratio of 0.8 and lr 0.001, under drop out = 0.5) we examined the acc:
| Dim |   acc |   epoch |    lr |   batch |   valid_ratio |
|----:|------:|--------:|------:|--------:|--------------:|
|  50 | 82.33 |       5 | 0.001 |      64 |          0.8  | 
| 300 | 84.77 |       5 | 0.001 |      64 |          0.8  | <<--- Selected

In order to examined the choice of drop out ratio, under consistent setting (Note we select 5 epoch, batch size of 
64 with validation ratio of 0.9 and lr 0.001) we examined the acc:

|  Drop out  |   acc |   epoch |    lr |   batch |   valid_ratio |
|-----------:|------:|--------:|------:|--------:|--------------:|
|          0 | 84.40 |       5 | 0.001 |      64 |          0.8  |
|       0.25 | 84.32 |       5 | 0.001 |      64 |          0.8  | 
|        0.5 | 84.75 |       5 | 0.001 |      64 |          0.8  | <<--- Selected
|       0.75 | 84.43 |       5 | 0.001 |      64 |          0.8  |

In order to see influence of multilayer vs single layer, under consistent setting (Note we select batch size of 
64 with validation ratio of 0.8 and lr 0.001) we examined the acc:

|         |   acc |   epoch |    lr |   batch |   valid_ratio |
|--------:|------:|--------:|------:|--------:|--------------:|
|  multi  | 84.47 |       5 | 0.001 |      64 |          0.8  | 
|  multi  | 84.65 |      10 | 0.001 |      64 |          0.8  | (We also compared against epoch of 10, considering linear layer might take few epochs to update) 
|  single | 84.71 |       5 | 0.001 |      64 |          0.8  | <<--- Selected

Training Parameters: 

In order to examine the optimized setting for the selected above settings, 4 level nested loop is conducted in iterating 
over all possible setting, with a total of 191 runs (each around 10-30 minutes under GPU):

lr_list = [0.0001, 0.001, 0.01, 0.1]
epochs = [5, 10, 20] #We tested via observing cross epoch comparison, it is observed any increase doesn't help
batch_size_list = [8, 16, 32, 64]
valid_train_ratio_list = [0.9, 0.8, 0.75, 0.5]

The top 10 is listed as the following:

|    |   acc |   epoch |    lr |   batch |   valid_ratio |
|---:|------:|--------:|------:|--------:|--------------:|
|  0 | 85.58 |       5 | 0.001 |      64 |          0.9  | <<--- Selected (64 Batch is faster comparing to batch size of 8, however there is trade offs)
|  1 | 85.48 |       5 | 0.001 |       8 |          0.9  |
|  2 | 85.3  |      10 | 0.001 |       8 |          0.8  |
|  3 | 85.17 |       5 | 0.001 |      32 |          0.9  |
|  4 | 85.17 |       5 | 0.001 |      16 |          0.75 |
|  5 | 85.13 |       5 | 0.001 |      32 |          0.8  |
|  6 | 85.04 |      10 | 0.001 |       8 |          0.9  |
|  7 | 85.04 |       5 | 0.001 |      16 |          0.8  |
|  8 | 85.03 |       5 | 0.001 |      16 |          0.9  |
|  9 | 84.96 |      20 | 0.001 |      16 |          0.9  |

(Note: we tested only slight higher epoch for multi layer, but improvement isn't shown hence discarded)

As researched, the choices of Epoch, Lr, Batch size and Valid Ratio highly depends on choices of network and data,
hence from our network epoch of 5, with lr 0.001 and batch size of 64 under valid_ratio of 0.9 to be optimal. The following
script is set up according to the optimal. 

(Note: Additional comments are placed when research is needed and conducted on the side of respective line)

"""

import torch
import torch.nn.functional as F
import re

import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # TODO: Note order is important, based research: https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95

    # Removing url
    sample = [re.sub(r'http\S+', '', x) for x in sample]

    # Removing Special Characters - For English Only
    sample = [re.sub(r"[^a-zA-Z]", "", x) for x in sample]

    # Converting all to lower case
    sample = [str(x).lower() for x in sample]

    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

# From Github List -
stopWords = {"a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"}
#stopWords = {}
wordVectors = GloVe(name='6B', dim=300) # Changed to 300 from 50


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # Converting the predicted rating output to values of either 0 or 1
    # using torch.rounding and change the tensor type of ratingOutput to LongTensor
    ratingOutput = torch.round(ratingOutput)
    ratingOutput = ratingOutput.type(torch.LongTensor)

    # From the 5 predicted outputs from the linear layer
    # we are only interested with the one that was predicted with the highest probability
    # so we use torch.max and .indices to locate the particular category that was predicted
    # with the highest probability.
    categoryOutput = torch.max(categoryOutput, 1)
    categoryOutput = categoryOutput.indices

    # Sending the prediction results to GPU
    categoryOutput = categoryOutput.to(device)
    ratingOutput = ratingOutput.to(device)

    return ratingOutput, categoryOutput


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

        # Parameters
        self.input_size = 300 # TODO: Adjustable - need to be same as vector Dim, tested 50 vs 300
        self.hidden_size = 256  # TODO: Adjustable - We can decrease or increase selected 256 based on research
        self.linear_layer_1_size = self.hidden_size * 2 # TODO: Decision in linear input after LSTM
        """
        # Extra layer - Better to be based on hidden size rather than a fix number?
        self.linear_layer_2_size = self.hidden_size
        self.linear_layer_3_size = int(self.hidden_size * (2 / 3))
        self.linear_layer_4_size = int(self.linear_layer_3_size / 5)
        """
        self.rating_out_size = 1  # TODO: Why 1 output node not 2? See: https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
        self.num_layers = 2 # TODO: Adjustable, 2 provide sufficient See: https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046#:~:text=Generally%2C%202%20layers%20have%20shown,to%20find%20reasonably%20complex%20features.
        self.category_out_size = 5
        self.drop_out_ratio = 0.5 # TODO: Adjustable - Tested, 0.25 did not work better,

        # Setting up the Bidirectional-LSTM network with input_size of 50, hidden_size of 256
        # and 2 layers
        self.lstm_network = tnn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Setting up two separate linear layer for predicting the category and ratings
        # and also the drop out step.
        # Category ------------------------------------------------------------------
        self.category_fc_1 = tnn.Linear(self.linear_layer_1_size, self.category_out_size) #Change if multi
        """
        # Updated with multi-layer - remember to fix hidden size if using
        self.category_fc_2 = tnn.Linear(self.linear_layer_2_size, self.linear_layer_3_size)
        self.category_fc_3 = tnn.Linear(self.linear_layer_3_size, self.linear_layer_4_size)
        self.category_fc_4 = tnn.Linear(self.linear_layer_4_size, self.category_out_size)
        """
        # Rating  ------------------------------------------------------------------
        self.rating_fc_1 = tnn.Linear(self.linear_layer_1_size, self.rating_out_size)
        """
        # Updated with multi-layer - remember to fix hidden size if using
        self.rating_fc_2 = tnn.Linear(self.linear_layer_2_size, self.linear_layer_3_size)
        self.rating_fc_3 = tnn.Linear(self.linear_layer_3_size, self.linear_layer_4_size)
        self.rating_fc_4 = tnn.Linear(self.linear_layer_4_size, self.rating_out_size)
        """
        # Drop-Out
        self.dropout = tnn.Dropout(self.drop_out_ratio)

    def forward(self, input, length):
        # Added Due to error:
        length = torch.as_tensor(length, dtype=torch.int64, device='cpu')  # Converting to CPU due to error in pack_padded_sequence

        # Pad each sequence in the batch so that they’re all the same length.
        packed_embedded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        # pass this batch of padded sequences through the lstm model.
        packed_output, (hn, cn) = self.lstm_network(packed_embedded)

        """
        # TODO: If you pass it to pack_padded, you can unpack via - but odd original method (using hn and cn) has better output
        # Weighted score: 67.72
        unpacked_out, _ = tnn.utils.rnn.pad_packed_sequence(packed_output)
        lstm_raw_out = unpacked_out.contiguous()
        output_r = lstm_raw_out[-1, :, :]
        output_c = lstm_raw_out[-1, :, :]
        """
        # Take the lstm hidden states and concatenate the forward and reversed sequences.
        output_r = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        output_c = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        """
        # applying a drop out step to regularise the output
        output_r = self.dropout(output_r)
        # the concatenated vector is passed to the final fully-connected linear layer
        # to make the ratings predictions.
        ratingOutput = self.rating_fc_1(output_r)
        # Initalise and apply the sigmoid activation function on the predicted ratings
        ratingOutput = F.sigmoid(ratingOutput)
        ratingOutput = ratingOutput.squeeze()
        """

        # Rating
        ratingOutput = self.dropout(output_r)
        ratingOutput = torch.sigmoid(self.rating_fc_1(ratingOutput))  # Output Layer - Using Sigmoid if 1 node output
        """
        # Attempt with multi-layer - remember to fix hidden size if using - No improvement saw
        ratingOutput = torch.tanh(self.rating_fc_1(ratingOutput)) #Layer 1
        ratingOutput = self.dropout(ratingOutput)
        ratingOutput = torch.tanh(self.rating_fc_2(ratingOutput)) #Layer 2
        ratingOutput = self.dropout(ratingOutput)
        ratingOutput = torch.tanh(self.rating_fc_3(ratingOutput)) #Layer 3
        ratingOutput = self.dropout(ratingOutput)
        ratingOutput = torch.sigmoid(self.rating_fc_4(ratingOutput)) #Output Layer - Using Sigmoid if 1 node output
        """
        ratingOutput = ratingOutput.squeeze()

        # Category
        categoryOutput = self.dropout(output_c)
        categoryOutput = torch.log_softmax(self.category_fc_1(categoryOutput),dim=1)
        """
        # Updated with multi-layer - remember to fix hidden size if using
        categoryOutput = torch.tanh(self.category_fc_1(categoryOutput))
        categoryOutput = self.dropout(categoryOutput)
        categoryOutput = torch.tanh(self.category_fc_2(categoryOutput))
        categoryOutput = self.dropout(categoryOutput)
        categoryOutput = torch.tanh(self.category_fc_3(categoryOutput))
        categoryOutput = self.dropout(categoryOutput)
        categoryOutput = torch.log_softmax(self.category_fc_4(categoryOutput), dim=1) # TODO: still think it requires activation here - it doesnt seem to affect regardless of choice, since you then use max
        """
        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # Converting the ratingTarget to float 32 as expected by the BCELoss() function
        ratingTarget = ratingTarget.type(torch.float32)

        # Calculating loss for ratings
        rating_loss = F.binary_cross_entropy(ratingOutput, ratingTarget)

        # Calculating loss for category
        category_loss = F.cross_entropy(categoryOutput, categoryTarget)

        # Add both losses together
        total_loss = rating_loss + category_loss

        return total_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################
trainValSplit = 0.9# TODO: Examined valid_train_ratio_list = [0.9, 0.8, 0.75, 0.5]
batchSize = 64# TODO: Examined batch_size_list = [8, 16, 32, 64]
epochs = 5# TODO: Examined epochs = [5, 10, 20]
# TODO: Why Adam is better? See: https://arxiv.org/pdf/1905.11286.pdf#:~:text=SGD%20with%20momentum%20is%20the,initialization%20and%20learn%2D%20ing%20rate.
optimiser = toptim.Adam(net.parameters(), lr=0.001) # TODO: Examined lr_list = [0.0001, 0.001, 0.01, 0.1]
