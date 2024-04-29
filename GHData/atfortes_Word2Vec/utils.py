import torch
import pandas as pd
from scipy import spatial
from constants import DEVICE
from scipy.stats import spearmanr
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def read_wordsim():
    df = pd.read_csv("wordsim353/combined.csv")
    words = df["Word 1"].to_list()
    ctx = df["Word 2"].to_list()
    scores = df["Human (mean)"].tolist()
    return words, ctx, scores


def plot_embeddings(model, dataset, words):
    pca = PCA(n_components=2)
    word_ids = [dataset.word2idx[w] for w in words if w in dataset.word2idx]
    print(word_ids)
    word_embeds = [model.center_embeds(torch.LongTensor([i]).to(DEVICE))[0].cpu().detach().numpy() for i in word_ids]
    embeds = pca.fit_transform(word_embeds)
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(embeds[:, 0], embeds[:, 1], ".")
    for i in range(len(embeds)):
        plt.annotate(words[i], xy=(embeds[i][0]-0.1, embeds[i][1]-0.1), fontsize=11)
    plt.show()


def calc_spearman(filename, model, dataset):
    ranks = []
    human = []
    df = pd.read_csv(filename)
    words1 = df["Word 1"].to_list()
    words2 = df["Word 2"].to_list()
    scores = df["Human (mean)"].tolist()
    embeddings = model.center_embeds
    for i in range(len(words1)):
        if words1[i] in dataset.word2idx and words2[i] in dataset.word2idx:
            word1_pos = dataset.word2idx[words1[i]]
            word2_pos = dataset.word2idx[words2[i]]
            word1_embed = embeddings(Variable(torch.LongTensor([word1_pos]).to(DEVICE))).cpu().detach().numpy().reshape(-1)
            word2_embed = embeddings(Variable(torch.LongTensor([word2_pos]).to(DEVICE))).cpu().detach().numpy().reshape(-1)
            cossim_score = 1 - spatial.distance.cosine(word1_embed, word2_embed)
            human.append(scores[i])
            ranks.append(cossim_score*10)
    df = pd.DataFrame(list(zip(human, ranks)))
    df.plot.scatter(0, 1, xlabel='Humans', ylabel='word2vec')
    return spearmanr(human, ranks).correlation
