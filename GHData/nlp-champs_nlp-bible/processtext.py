import pandas as pd

df = pd.read_csv('./t_kjv.csv'); # read in data from Kaggel

file = open('bible-text.txt', 'w') # open a file to write to

# for tracking current book / chapter
iCurrBook = 1
iCurrChapter = 1
file.write('BOOK ' + str(iCurrBook) + ', CHAPTER ' + str(iCurrChapter) + ':\n') # very first line
# loop through all rows
for index, row in df.iterrows():
    if iCurrBook == row['b'] and iCurrChapter == row['c']: # if we match both the current book and chapter
        file.write(row['t'] + '\n') # just write the text
    else: # else, its a new beginning! a NEWWW BEGINNING!!!!
        if iCurrBook != row['b']:
            iCurrBook = iCurrBook + 1 #book increment
            iCurrChapter = 1 # and reset the chapter to 1
        elif iCurrChapter != row['c']: # otherwise it is just a chapter increment
            iCurrChapter = iCurrChapter + 1
        file.write('\n'); # new line to seperate chapters
        file.write('BOOK ' + str(iCurrBook) + ', CHAPTER ' + str(iCurrChapter) + ':\n')
