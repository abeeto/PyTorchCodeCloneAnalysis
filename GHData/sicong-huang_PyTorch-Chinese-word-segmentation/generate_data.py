import re
import sys

path = sys.argv[1]

# read in file to be converted
f = open(path, 'r', encoding='utf8')
raw_doc = f.read()
f.close()

# remove space and write to output file
doc_no_space = re.sub(' ', '', raw_doc)
f = open(re.sub('\.txt', '_no_space.txt', path), 'w', encoding='utf8')
f.write(doc_no_space)
f.close()

# convert to 0s and 1s as label
doc_label = re.sub('. {1,3}', '1', raw_doc)
doc_label = re.sub('.\n', '1\n', doc_label)
doc_label = re.sub('[^1\n]', '0', doc_label)
f = open(re.sub('\.txt', '_label.txt', path), 'w', encoding='utf8')
f.write(doc_label)
f.close()
