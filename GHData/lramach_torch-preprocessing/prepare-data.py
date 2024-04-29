#!/usr/local/bin/python
import numpy as np
import pandas
import h5py
from pandas import HDFStore, DataFrame
import re

#Loading the essays in the train set
#filename - name of the file to load data from
#flag - to determine whether to load scores from file
def loadData(filename, flag):
	if flag == 1:
	  scores = np.loadtxt(filename, delimiter='\t', usecols=[1], dtype=np.float)
	  essays = np.loadtxt(filename, delimiter='\t', usecols=[2], dtype=np.str)
	  return(essays, scores)
        else:
	  essays = np.loadtxt(filename, delimiter='\t', usecols=[0], dtype=np.str)
          return essays

def generateVocabulary(train_data):
	#reading and getting a list of unique tokens
	wordslist = set() #initializing the array
        for line in train_data:
            line = line.lower() #lowercase string
	    line = re.sub('[^\w\s]','',line) #replace punctuations 
	    words = line.split(" ")
    	    for word in words:
                word = word.lower().replace("\n", "")
                if(word not in wordslist):
                    #wordslist contains the vocabulary of the training set
                    wordslist.add(word)
	return wordslist

def addResponseToksToVocab(reference_data, wordslist):
	reference_data = np.atleast_1d(reference_data) #to read arrays with just one item
	for line in reference_data:
	   line = line.lower()
	   line = re.sub('[^\w\s]','',line)
	   words = line.split(" ")
	   for word in words:
               word = word.lower().replace("\n", "")
	       if(word not in wordslist):
	           wordslist.add(word)
	return wordslist

#Loading scores of the train data
def getScores(filename):
	train_scores = np.loadtxt(filename, dtype = np.int)
        return train_scores

#Loading glove, reducing its size to that of the vocabulary
def loadingGlove(filename, outfile, wordslist):
	glovevecs = np.loadtxt(filename, delimiter=' ', dtype=np.float, usecols=range(1,301)) #exclude the first column
	glovewords = np.loadtxt(filename, delimiter=' ', dtype=np.str, usecols=range(0,1)) 
	#selecting a subset of words from the glovevec object
	glovemat = []
	selectedwords = []
	for i in range(0, len(glovewords)): 
    		if glovewords[i] in wordslist:
        		glovemat.append(glovevecs[i])
        		selectedwords.append(glovewords[i])
	glovemat = np.array(glovemat)#contains the float values of the word vectors
	#print("Size of the glovevec matrix:")
	#print(len(glovemat))
	#writing out the subset of the the glove vector matrix
	store = h5py.File(outfile, "w")
	store.create_dataset("glovevec", data=glovemat)
	store.close()
	return selectedwords

#Generating key-index pairs
def generateWordIndexPairs(selectedwords):
	wordIndex = np.array
	wordIndex = np.vstack((selectedwords, range(1, (len(selectedwords)+1)))).T 
	#generating the dictionary
	wordIndexDict = dict(list(wordIndex))
	return wordIndexDict

#Getting indices of the vocabulary terms, and creating vectors of indices for each of the data points
def getVectorsOfIndices(essays, wordIndexDict):
	essays_index = []
	essays = np.atleast_1d(essays)
	for essay in essays:
	    essay = essay.lower()
	    essay = re.sub('[^\w\s]','',essay)
    	    tokens = essay.split(" ")
            temp = []
            for token in tokens:
                if wordIndexDict.has_key(token): #if token is present in the dictionary
                    temp.append(int(wordIndexDict[token]))
            essays_index.append(temp)
	return essays_index

def writeVectorsOfIndices(filename, indices, scores = None):
	store = h5py.File(filename, "w")
	df = DataFrame(indices)
	df.fillna(-1, inplace =True)
	store.create_dataset("index", data=df, dtype=np.int)
	if scores is not None:
	  store.create_dataset("target", data=scores, dtype=np.int) 
	store.close()

#Load train file
DataPath = "path to data"
prompts = [""]
for prompt in prompts:
    print(prompt)
    output = loadData(DataPath +str(prompt)+"-trainSubset3.csv", flag = 1)
    train_essays = output[0] 
    train_scores = output[1] 
    #print(train_scores)
    #Load validation file
    output = loadData(DataPath+str(prompt)+"-valid.csv", flag = 1)
    validation_essays = output[0] 
    validation_scores = output[1] 
    #Load test file
    output = loadData(DataPath+str(prompt)+"-test.csv", flag = 1)
    test_essays = output[0] 
    test_scores = output[1]
    #Using train data to get vocabulary
    wordslist = generateVocabulary(train_essays)
    #print(len(wordslist))
    #load the human references file
    references = loadData(DataPath + str(prompt)+"-references.csv", flag = 0)
    wordslist = addResponseToksToVocab(references, wordslist)

    #Load glove vectors
    wordOrderInGlove = loadingGlove("path to glovefile.txt", "outputfile.h5", wordslist)

    # #Get Word-Index dictionary
    wordIndexDict = generateWordIndexPairs(wordOrderInGlove)
    
    # #Get index vectors for the train, validation and test sets
    train_essay_index = getVectorsOfIndices(train_essays, wordIndexDict)
    validation_essay_index = getVectorsOfIndices(validation_essays, wordIndexDict)
    test_essay_index = getVectorsOfIndices(test_essays, wordIndexDict)
    references_index = getVectorsOfIndices(references, wordIndexDict)
    
    # #Writing out the index files
    writeVectorsOfIndices("pathToFile"+ str(prompt) +"_train.h5", train_essay_index, train_scores)
    writeVectorsOfIndices("pathToFile"+ str(prompt) +"_validation.h5", validation_essay_index, validation_scores)
    writeVectorsOfIndices("pathToFile"+ str(prompt) +"_test.h5", test_essay_index, test_scores)
    writeVectorsOfIndices("pathToFile"+ str(prompt) +"_references.h5", references_index, None)
