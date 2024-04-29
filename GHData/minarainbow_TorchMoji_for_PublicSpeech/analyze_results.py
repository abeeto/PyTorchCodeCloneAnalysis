import csv 
import collections
import json
from collections import Counter
import sys


columns = collections.defaultdict(list) # each value in each column is appended to a list
emoji_scores = collections.defaultdict(int)# each value

with open(sys.argv[1]) as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

for emoji in columns['Pct_1']:
    emoji_scores[emoji] += 10
for emoji in columns['Pct_2']:
    emoji_scores[emoji] += 9
for emoji in columns['Pct_3']:
    emoji_scores[emoji] += 8
for emoji in columns['Pct_4']:
    emoji_scores[emoji] += 7
for emoji in columns['Pct_5']:
    emoji_scores[emoji] += 6
    
sorted_emoji_scores = collections.OrderedDict(sorted(emoji_scores.items(), key=lambda kv: kv[1], reverse=True))
labels = sorted_emoji_scores.keys()
scores = sorted_emoji_scores.values()
print(sorted_emoji_scores)