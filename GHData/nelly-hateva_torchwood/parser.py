import glob
import os
import re
import string
from bs4 import BeautifulSoup
from stemmer import PorterStemmer


here = lambda *x: os.path.abspath(os.path.join(os.path.dirname(__file__), *x))
data_dir = here('data', )
corpus_dir = here('corpus', )

total_files = 0
p = PorterStemmer()

if not os.path.exists(corpus_dir):
    os.makedirs(corpus_dir)
os.chdir(data_dir)

for file in glob.glob('*.sgm'):
    current_file = os.path.join(data_dir, file)
    print 'Extract files from file %s' % current_file
    soup = BeautifulSoup(open(current_file))
    for document in soup.find_all('reuters'):
        new_file = os.path.join(corpus_dir, document.get('newid'))
        with open(new_file, "wb") as extracted_file:
            read_data = document.get_text().encode('utf-8')
            clean_data = re.sub(r'/', ' / ', read_data)
            clean_data = re.sub(r'-', ' - ', clean_data)
            """
            The punctuations contained in the string.punctuation are
            !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            """
            clean_data = read_data.translate(None, string.punctuation)
            clean_data = clean_data.lower()
            output = ''
            for word in clean_data.split():
                output += p.stem(word, 0, len(word) - 1)
                output += ' '
            extracted_file.write(output)
            total_files += 1
print 'Total files extracted %s' % total_files
