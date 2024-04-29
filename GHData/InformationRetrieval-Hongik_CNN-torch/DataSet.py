import pickle as pkl
import numpy as np
import json

import pandas as pd
import nltk
from konlpy.tag import Okt

from keras.preprocessing.sequence import pad_sequences

okt = Okt()


def tokenize(doc):
    """
    input:
        doc: dtype=string, example) '아 더빙.. 진짜 짜증나네요 목소리'
    return:
        list: dtype=string, elements example) ['아/Exclamation', '더빙/Noun', '../Punctuation', '진짜/Noun', '짜증나다/Adjective', '목소리/Noun']
    """
    return ["/".join(token) for token in okt.pos(doc, norm=True, stem=True)]


def saveDocs(docs, filePath):
    # remove duplicate samples.
    docs.drop_duplicates(subset=["document"], inplace=True)

    # remove all characters except korean and spaces.
    docs["document"] = docs["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # exchange spaces value to null value.
    docs["document"].replace("", np.nan, inplace=True)

    # remove null samples.
    docs = docs.dropna(how="any")

    # reviewDocs has [("morpheme1/tag1", "morpheme2/tag2", ... ,), label(0 or 1)]
    reviewDocs = [(tokenize(row[1]), row[2]) for row in docs.values]

    with open(filePath, "wb") as f:
        pkl.dump(reviewDocs, f)

def SaveDataPkl(x, filepath):
    with open(filepath, "wb") as f:
        pkl.dump(x, f)
        
def loadDocs(filePath):
    with open(filePath, "rb") as f:
        reviewDocs = pkl.load(f)
        return reviewDocs


        
def frequency2index(jsonDict, topN):
    lookupTable = dict()
    lookupTable["<pad>"] = 0  # pad value to match the maximum length.
    lookupTable["<oov>"] = 1  # out of value (if a word not in lookup table's keys.)
    i = 2
    for key in jsonDict.keys():
        lookupTable[key] = i
        i += 1

    return lookupTable


def saveTopNwords(dataTokens, topN, filePath):
    text = nltk.Text(dataTokens, name="NMSC")

    # get TOP N tokens with the highest frequency of output.
    topN_words = {token[0]: token[1] for token in text.vocab().most_common(topN)}

    with open(filePath, "w", encoding="utf-8") as jsonF:
        json.dump(topN_words, jsonF, ensure_ascii=False)


def loadTopNwords(filePath):
    with open(filePath, "r", encoding="utf-8") as jsonF:
        topN_words = json.load(jsonF)

    return topN_words


def saveTopNindex(topN_words, topN, filePath):
    topN_index = frequency2index(topN_words, topN=topN)
    with open(filePath, "w", encoding="utf-8") as jsonF:
        json.dump(topN_index, jsonF, ensure_ascii=False)


def loadTopNindex(filePath):
    with open(filePath, "r", encoding="utf-8") as jsonF:
        topN_index = json.load(jsonF)
        return topN_index


def encodeToInt(train_x, lookupTable, maxLen):
    encodeIntList = []
    keySet = lookupTable.keys()
    for x in train_x:
        encodeIntLine = []
        for token in x:
            if token in keySet:
                encodeIntLine.append(lookupTable[token])
            else:
                encodeIntLine.append(lookupTable["<oov>"])
        encodeIntList.append(encodeIntLine)

    encodeIntList = np.array(encodeIntList, dtype="object")
    padEncodeIntList = pad_sequences(encodeIntList, maxlen=maxLen, padding="pre", value=lookupTable["<pad>"])
    return padEncodeIntList

    for xIdx in range(len(train_x)):
        for tokenIdx in range(len(train_x[xIdx])):
            if train_x[xIdx][tokenIdx] in keySet:
                token = lookupTable[train_x[xIdx][tokenIdx]]
            else:
                token = lookupTable["<oov>"]


if __name__ == "__main__":
    # train_df = pd.read_csv("./nsmc-master/ratings_train.txt", "\t")
    # test_df = pd.read_csv("./nsmc-master/ratings_test.txt", "\t")
    # data_df = pd.read_table("./nsmc-master/ratings.txt")

    # saveDocs(train_df, "./trainDocs.pkl")
    # saveDocs(test_df, "./testDocs.pkl")
    # saveDocs(data_df, "./dataDocs.pkl")

    # trainDocs = loadDocs("./trainDocs.pkl")
    # testDocs = loadDocs("./testDocs.pkl")

    # ==================================================================================================
    # ========= process that get a Top N lookup table from comprehensive data set[ratings.txt] =========
    # ==================================================================================================
    if False:
        dataDocs = loadDocs("./dataDocs.pkl")

        # get all tokens from dataDocs.
        dataTokens = [token for doc in dataDocs for token in doc[0]]

        # get TOP N tokens with the highest frequency of output.
        topN = 10000
        saveTopNwords(dataTokens=dataTokens, topN=topN, filePath="./top_%d_words.json" % (topN))
        topN_words = loadTopNwords("./top_%d_words.json" % (topN))

        # convert frequency words dictionary to lookup table.
        saveTopNindex(topN_words=topN_words, topN=topN, filePath="./top_%d_index.json" % (topN))
        topN_index = loadTopNindex("./top_%d_index.json" % (topN))

    # ==================================================================================================
    # === process that encode train data[ratings_train.txt] to integer data type(lookup table index) ===
    # ==================================================================================================

    trainDocs = loadDocs("./trainDocs.pkl")
    print("train Docs list length :", len(trainDocs))

    # divide by input x and label y
    train_x = []
    train_y = []
    for x, y in trainDocs:
        train_x.append(x)
        train_y.append(y)

    maxLen = 80
    lookupTable = loadTopNindex("./top_10000_index.json")
    train_x = encodeToInt(train_x, lookupTable, maxLen)  # get numpy array that converted from morpheme to interger(lookup table index value)
    train_y = np.array(train_y)

    # ==================================================================================================
    # ==== process that encode test data[ratings_test.txt] to integer data type(lookup table index) ====
    # ==================================================================================================

    testDocs = loadDocs("./testDocs.pkl")
    print("test Docs list length :", len(testDocs))

    # divide by input x and label y
    test_x = []
    test_y = []
    for x, y in testDocs:
        test_x.append(x)
        test_y.append(y)

    maxLen = 80
    lookupTable = loadTopNindex("./top_10000_index.json")
    test_x = encodeToInt(test_x, lookupTable, maxLen)  # get numpy array that converted from morpheme to interger(lookup table index value)
    test_y = np.array(test_y)
    
    SaveDataPkl(train_x, "./Data/train_x.pkl")
    SaveDataPkl(train_y, "./Data/train_y.pkl")
    SaveDataPkl(test_x, "./Data/test_x.pkl")
    SaveDataPkl(test_y, "./Data/test_y.pkl")
    SaveDataPkl(lookupTable, "./Data/lookupTable.pkl")