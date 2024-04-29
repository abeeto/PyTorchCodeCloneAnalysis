# This is the server script that runs on the docker image. 
# You don't need to execute it, but you can modify it easier this way.
# Still don't forget after modifying this script to upload it to the server and commit to the docker image. 

import subprocess
import socket
from socket import error as SocketError
import errno

import os
import subprocess
from subprocess import Popen, PIPE, STDOUT
import time

import json
import pronouncing
from random import randint

from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data

import sys
import numpy as np

import shutil

import enchant
from cStringIO import StringIO

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
host = '0.0.0.0' # Get local machine name
port = 12345              # Reserve a port for your service.
s.bind(('', port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
print 'listening on port 12345'
while True:
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   c.send('Thank you for connecting')
      # receive image file
   i=1
   f = open('claycam'+ str(i)+".jpg",'wb')
   i=i+1
   # wait for densecap & rnn lib to handwrite haiku and create image
      # receive and write image
   l = c.recv(4096)
   try:
   	while (l): 
      		f.write(l)
      		l = c.recv(4096)
   except socket.error:
   	print "image received"
   	break
f.close()

# start densecap neural network to generate captions (and write them in json-file)
os.chdir('/root/densecap')
subprocess.call('/root/torch/install/bin/th run_model.lua -input_image /root/claycam1.jpg -gpu -1', shell=True)
print "analyse image and seed torch-rnn"
# get the data from the json file
clay_base = open('/root/densecap/vis/data/results.json')
wjson = clay_base.read()
wjdata = json.loads(wjson)
wjdata_list = wjdata['results'][0]['captions']

# create empty storage for selected captions with fitting syllables (with either 5 or 7 syllables)
syllables5 = []
syllables7 = []
syllables23 = []

# check all captions for fitting syllables (using pronouncingpy + CMU pronouncing dictionary)
# add them to the empty storage
for i in range (1, 83):

   try:
      text = wjdata['results'][0]['captions'][i - 1]

      phones = [pronouncing.phones_for_word(p)[0] for p in text.split()]
      count = sum([pronouncing.syllable_count(p) for p in phones])
      for y in range (1, 2):
         if int(count) == 5:
            syllables5.append(wjdata['results'][0]['captions'][i - 1])
      for x in range (0, 1):
         if int(count) == 7:
            syllables7.append(wjdata['results'][0]['captions'][i - 1])
      for z in range (0, 1):
         if int(count) == 3 or int(count) == 2:
            syllables23.append(wjdata['results'][0]['captions'][i - 1])

# skip over errors caused by non-indexed word <UNK> in captions
   except IndexError:
         pass
   continue

# create arrays for pre-selections of fitting syllables
selection_line1 = ['fill']
selection_line2 = ['fill']
selection_line3 = ['fill']

# load the pretrained neural net for checking the haiku-lines
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# create storage for checking nouns
check_noun = ['noun', 'noun', 'noun']
check_verb = []
print "selecting haiku lines"

# randomise selection per syllable selection and avoid repetition in words
while len(check_noun) != len(set(check_noun)):
    check_noun = []
    check_verb = []
    selection_line1 = syllables5 [randint(0,(len(syllables5) -1) /2)]
    selection_line2 = syllables7 [randint(0,(len(syllables7)-1))]
    selection_line3 = syllables5 [randint(len(syllables5)/2,(len(syllables5)-1))]
    # create strings for further language processing
    selection_line1 = ''.join(selection_line1)
    selection_line2 = ''.join(selection_line2)
    selection_line2 = ''.join(selection_line2)

    # tokenize the text
    tokenized = tokenizer.tokenize(selection_line1)
    words1 = word_tokenize(selection_line1)

    tokenized = tokenizer.tokenize(selection_line2)
    words2 = word_tokenize(selection_line2)

    tokenized = tokenizer.tokenize(selection_line3)
    words3 = word_tokenize(selection_line3)

    # identify the parts of speech
    tagged1 = nltk.pos_tag(words1)
    tagged2 = nltk.pos_tag(words2)
    tagged3 = nltk.pos_tag(words3)

    try:
		for i in range(0,len(words1)):
        	# find nouns and verbs in line1 and store them
        		if tagged1[i][1] == 'NN':
                		check_noun.append(tagged1[i])
        		if tagged1[i][1] == 'NNS':
                		check_noun.append(tagged1[i])       
        		if tagged1[i][1] == 'VBG':
                		check_verb.append(tagged1[i])
        		if tagged1[i][1] == 'VBZ':
                		check_verb.append(tagged1[i])

   	 	for i in range(0,len(words2)):
        		# find nouns and verbs in line2 and store them
        		if tagged2[i][1] == 'NN':
                		check_noun.append(tagged2[i])
        		if tagged2[i][1] == 'NNS':
                		check_noun.append(tagged2[i])        
        		if tagged2[i][1] == 'VBG':
                		check_verb.append(tagged1[i])
        		if tagged2[i][1] == 'VBZ':
                		check_verb.append(tagged1[i])
	
    		for i in range(0,len(words3)):
        		# find nouns and verbs in line3 and store them
        		if tagged3[i][1] == 'NN':
                		check_noun.append(tagged3[i])
        		if tagged3[i][1] == 'NNS':
                		check_noun.append(tagged3[i])
        		if tagged3[i][1] == 'VBG':
                		check_verb.append(tagged1[i])
        		if tagged3[i][1] == 'VBZ':
                		check_verb.append(tagged1[i])
    except IndexError:
	pass
		
    # check for existing lines with verbs 
    if not check_verb:
        check_noun = ['noun', 'noun', 'noun']

    # check for repetitions at the beginning of lines
    if words1[0] == words2[0] or words2[0] == words3[0]:
        check_noun = ['noun', 'noun', 'noun']

    word_check = selection_line1 + " " + selection_line2 + " " + selection_line3
    #print word_check
    # check for word "background" which messes up rnnlib
    if "background" in word_check:
        check_noun = ['noun', 'noun', 'noun']
    if "brown" or "Brown" in word_check:
      selection_line1 = selection_line1.replace("brown", "")
      selection_line1 = selection_line1.replace("Brown", "")
      selection_line2 = selection_line2.replace("brown", "")
      selection_line2 = selection_line2.replace("Brown", "")
      selection_line3 = selection_line3.replace("brown", "")
      selection_line3 = selection_line3.replace("Brown", "")

# return the checked lines
print "clay-lines selected:"
word_check = selection_line1 + ", " + selection_line2 + ", " + selection_line3
print (word_check)

# generate quotes with torch-rnn trained on metaphysics philosophy
print "creating quotes"
os.chdir('/root/torch-rnn')

d = enchant.Dict("en_US")
e = enchant.Dict("en_GB")
sen = None
while sen is None:
  proc = subprocess.Popen('th sample.lua -checkpoint cv/checkpoint_140400.t7 -length 1000 -gpu -1 -start_text "%s?" -temperature 0.7' %(word_check), shell=True, stdout=subprocess.PIPE)
  nu = proc.stdout.read()
  print nu
  sents = nu.split('?')
  sents = ''.join(sents[1])
  sents = sents.split('.')
  # pick sentence that is shorter than 10 words, check if the words in sentence exist and stop the generation of quotes
  for x in sents[1:-1]:
        if len(x.split()) < 10:
            counter = 0
            for word in x.split():
                if d.check(word) or e.check(word) is "TRUE":
                    counter = counter + 1
            if counter == len(x.split()):
                if len(x.split()) > 2:
                    sen = ' '.join(x.split())
                    print "%s." % sen
                    break
quote = open("quote.txt", "w")
quote.write("%s." % sen)
quote.close()
# send quote back to client
s.listen(5)                
print 'listening on port 12345'
c, addr = s.accept()    
print 'Got connection from', addr
print "quote generated: %s" %(sen)
f=open ("/root/torch-rnn/quote.txt", "rb")
l = f.read(1024)
while (l):
	c.send(l)
        l = f.read(1024)
print "quote sent"
f.close()
c.send('done')
print c.recv(1024)
i = c.recv(1024)
if "file received" in i:
	print "closing connection"
	time.sleep(1)
	c.close()                # Close the connection
	time.sleep(5)
