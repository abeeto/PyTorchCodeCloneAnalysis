#!/usr/local/bin/python
import numpy as np
import pandas
import h5py
from pandas import HDFStore, DataFrame
import re
from collections import OrderedDict

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

def generateVocabulary(train_data, n):
	#reading and getting a list of unique tokens
        char_counts = {}
	for line in train_data:
            line = line.lower() #lowercase string
	    line = re.sub('[^\w\s]','',line) #replace punctuations 
	    #get character 3-grams
	    chargrams = []
	    for x in range(len(line)):
		charGr = line[x:x+n]
		if len(charGr) == n:
	 	          chargrams.append(charGr)
    	    for char in chargrams:
                char = char.lower().replace("\n", "")
                if(char not in char_counts):
		    char_counts[char] = 1.0 #First occurrence of the char-gram
                char_counts[char] += 1.0 #Increment existing count
    
        #Sorting the char-grams by frequency
        sorted_chars = sorted(char_counts.keys(), key=lambda x: char_counts[x], reverse=True)
	select_list = sorted_chars[:1000] #Selecting top 1000
	sorted_chars = sorted(select_list, key=str.lower) #Sorting alphabetically, doing this to check
        return sorted_chars 

#Loading scores of the train data
def getScores(filename):
	train_scores = np.loadtxt(filename, dtype = np.int)
        return train_scores

#Getting indices of the vocabulary terms, and creating vectors of indices for each of the data points
def getCharVectors(essays, chars_list, n):
	essays_vectors = []
	essays = np.atleast_1d(essays)
	for essay in essays:
	    essay = essay.lower()
	    essay = re.sub('[^\w\s]','',essay)
            #get character n-grams
	    chargrams = []
	    for x in range(len(essay)):
	        charGr = essay[x:x+n]
		if len(charGr) == n:
		    chargrams.append(charGr)

	    char_vector = {}
	    #initialize vector
	    for char in chars_list:
	        char_vector[char] = 0.0 #initialize to 0

	    #iterating over char grams in essay and incrementing vector count 
            for char in chargrams:
		if char in chars_list: #making sure the char gram is in the train vocab
		    if char in char_vector:
	                char_vector[char] += 1.0 #increment count
           	    char_vector[char] = 1.0 #initialize, seen item once

	    char_vector = OrderedDict(sorted(char_vector.items(), key = lambda t:t[0])) #sort by keys before adding to array of char vectors
	    #OrderedDict maintains the order of the values in the sorted hash, a regular dictionary does not maintain order
	    #print(char_vector)
	
	    #Adding the counts to the array of vectors
	    essays_vectors.append(char_vector.values())
	return essays_vectors

def writeVectors(filename, chars, scores = None):
	store = h5py.File(filename, "w")
	df = DataFrame(chars)
	df.fillna(-1, inplace =True)
	store.create_dataset("chars", data=df, dtype=np.int)
	if scores is not None:
	  store.create_dataset("target", data=scores, dtype=np.int) 
	store.close()

#Load train file
DataPath = "path to data"
prompts = [""]
for prompt in prompts:
    print(prompt)
    output = loadData(DataPath +str(prompt)+"-train.csv", flag = 1)
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
    #Using train data to get vocabulary from the train_essays and n-gram
    charslist = generateVocabulary(train_essays, n = 3)
    print(charslist)
    
    # Generate char-gram vectors of frequencies
    train_essays_chars = getCharVectors(train_essays, charslist, 3)  
    validation_essays_chars = getCharVectors(validation_essays, charslist, 3)  
    test_essays_chars = getCharVectors(test_essays, charslist, 3)  

    #Writing out the character vector files
    writeVectors("pathToFile"+ str(prompt) +"_train_char3gram.h5", train_essays_chars, train_scores)
    writeVectors("pathToFile"+ str(prompt) +"_validation_char3gram.h5", validation_essays_chars, validation_scores)
    writeVectors("pathToFile"+ str(prompt) +"_test_char3gram.h5", test_essays_chars, test_scores)
