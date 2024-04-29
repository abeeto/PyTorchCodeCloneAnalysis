import numpy, torch, csv, pandas

with open('PridePrejudice2.txt') as f:
    text = f.read()

lines = text.split('\n')
line = lines[200]
line

letterTensor = torch.zeros(len(line), 128)
letterTensor.shape

for i, letter in enumerate(line.lower().strip()):
    letterIndex = ord(letter) if ord(letter) < 128 else 0
    letterTensor[i][letterIndex] = 1


def cleanWords(inputStr):
    punctuation = '.,;:"!?”“_-'
    wordList = inputStr.lower().replace('\n',' ').split()
    wordList = [word.strip(punctuation) for word in wordList]
    return wordList

wordsInLine = cleanWords(line)
line, wordsInLine

wordList = sorted(set(cleanWords(text)))
wordToIndexDict = {word: i for (i, word) in enumerate(wordList)}

len(wordToIndexDict), wordToIndexDict['impossible']

wordTensor = torch.zeros(len(wordsInLine), len(wordToIndexDict)).cuda()
for i, word in enumerate(wordsInLine):
    wordIndex = wordToIndexDict[word]
    wordTensor[i][wordIndex] = 1
    print('{:2} {:4} {}'.format(i, wordIndex, word))

