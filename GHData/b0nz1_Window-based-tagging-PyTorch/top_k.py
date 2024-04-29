import numpy as np
import heapq
"""
Igal Zaidman 311758866
Yehuda Gihasi 305420671
"""
def most_similar(word, k):
    w_v = vocab[word]
    distances = {} 
    
    for i,v in enumerate(vecs):
        if word != vocabWords[i]:
            curr_dist = np.dot(w_v,v)/(np.sqrt(np.dot(w_v,w_v))*np.sqrt(np.dot(v,v)))
            distances[curr_dist] = vocabWords[i]
    
    top_vals = heapq.nlargest(k, distances)
    top_words = [distances[key] for key in top_vals]
    return top_words, top_vals

if __name__ == "__main__":
    vecs = np.loadtxt("wordVectors.txt")
    vocabWords = open("vocab.txt", "r").read().lower().split('\n')
    with open('vocab.txt', 'r') as f:
        line = [w.strip() for w in f.readlines()]
        vocab = {w: vec for w, vec in zip(line, vecs)}
    
    list_of_words = ["dog","england","john","explode","office"]
    for w in list_of_words:
        print(most_similar(w, 5))
