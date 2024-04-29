from tools.DataSet.SNIPS import snips
import os.path as op
from tools.conll import evaluate
def getRawSentences(path):
    rawText = []
    sentence = []
    label = []
    intent = []

    for line in open(path, 'r'):
        row = line.strip().split()

        # read the blank space
        if len(row) == 0:
            rawText.append([sentence.copy(), label.copy(), intent.copy()])
            sentence.clear()
            label.clear()

        # read the intent
        elif len(row) == 1:
            intent = row

        # read the word and label
        elif len(row) == 2:
            sentence.append(row[0])
            label.append(row[1])

    rawText.append([sentence.copy(), label.copy(), intent.copy()])

    return rawText

# data = getRawSentences('/home/sh/data/JointSLU-DataSet/formal_snips/test.txt')
#
# with open('rawtest.txt', 'w') as f:
#     for i in data:
#         f.write(str(i[0])+'\n')
#         f.write(str(i[1])+'\n')
#         f.write(str(i[2])+"\n")
#         f.write('\n')

print(evaluate(['B-shit', 'I-shit', 'O', 'O', 'B-a', 'I-a', 'I-a'],['B-shit', 'I-shit', 'O', 'B-c', 'O', 'B-b', 'I-b']))

