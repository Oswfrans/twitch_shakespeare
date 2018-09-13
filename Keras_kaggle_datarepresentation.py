import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import gc

import numpy as np
from six.moves import urllib
from six.moves import xrange  #

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, to_categorical

from sklearn.model_selection import train_test_split
#https://shalabhsingh.github.io/Text-Generation-Word-Predict/
#https://www.kaggle.com/mrisdal/intro-to-lstms-w-keras-gpu-for-text-generation/

train= True  
#I removed SP!!!!!!!!

#go higher first, 7k off the bat
vocabulary_size = 200 #3400
sentenceLength = 20
 #C:\\Users\\o.frans\\Python Scripts\\
with open('C:\\Users\\Oswin\\Desktop\\twitch-master\\SP.txt') as data_file:
    SP=data_file.read()

#quick and dirty use regex for a better implementation
SPwords = SP.replace('\n', ' ').split(' ')

#database.txt
with open('C:\\Users\\Oswin\\Desktop\\twitch-master\\database_half.txt') as data_file:
    twitch=data_file.read()

twitchwords = twitch.replace('<eos>', ' ').split(' ')

# Build the model 

######should add batchnormalization somewhere??
# model.add(BatchNormalization())
#####

#rethink structure and dimensions
# 2 to 4 layers

model = Sequential() 

#karpathy suggests 0.5 for dropout
#Use LSTM over GRU

model.add(LSTM(256, input_shape = (sentenceLength, vocabulary_size), return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(256))
model.add(Dropout(0.1))

model.add(Dense(vocabulary_size, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

print(model.summary()) 

#testing

#SPwords +
cents = twitchwords + SPwords
#potential problem, maybe remove?
cents = [s.lower() for s in cents if s!='']

print('Data size', len(cents))

# Step 2: Build the dictionary and replace rare words with UNK token.
 #80000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    cents, vocabulary_size)
#del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

del data
del count
del twitch
del SP

del SPwords

gc.collect()

sentences = []
next_word =[]

for i in range(0, len(cents) - sentenceLength, 1):
	sentences.append(cents[i:i+sentenceLength])
	next_word.append(cents[i+sentenceLength])

print("we have so many sentences: " + str(len(sentences)))
print("we have so many next words: " + str(len(next_word)))	

X = np.zeros((len(sentences), sentenceLength, vocabulary_size), dtype=np.bool)
Y = np.zeros((len(sentences), vocabulary_size), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        if word in dictionary:
          X[i, t, dictionary[word]] = 1
        else:
          X[i, t, 0] = 1
    if next_word[i] in dictionary:
      Y[i, dictionary[next_word[i]]] = 1
    else:
      Y[i, 0] = 1

print("no Error so far")
gc.collect()

#memory reducing
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 36)

gc.collect()

#they are just long lists
#print(len(X_train))

if train == True:
	model.fit(X, Y, epochs=5, batch_size= 64)
  #model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size= 64) 
	#save model
	model.save('C:\\Users\\Oswin\\Desktop\\twitch-master\\model_kaggle_2.h5')
else:
	#load model
	model = keras.models.load_model('C:\\Users\\Oswin\\Desktop\\twitch-master\\model_kaggle_2.h5')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).flatten() #.astype('float64')
    preds = np.log(preds) / temperature

    dist = np.exp(preds)/np.sum(np.exp(preds)) 
    choices = range(len(preds)) 
    return np.random.choice(choices, p=dist)

    #apperently there is a bug that breaks this
    #https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4
    #exp_preds = np.exp(preds)
    #preds = exp_preds / np.sum(exp_preds)
    #probas = np.random.multinomial(1, preds, 1)
    #return np.argmax(probas)

diversity= 1.1

#initial_text = 'kappa sir kappa king of england kappa residentsleeper sir shakespeare kappa hamlet is king kappa you shine like a beacon'
#initial_text = 'When forty winters shall besiege thy brow And dig deep trenches in thy beauty field Thy youth proud livery so' 
initial_text = 'deIlluminati only the VAPEST wizard can wear THIS HAT VapeNation deIlluminati only the VAPEST wizard can wear THIS HAT VapeNation'
initial_text = [dictionary[c] if c in dictionary else 0 for c in initial_text.split(' ')]
#initial_text= [21, 237, 21, 237, 21, 237, 0, 1, 100,  21, 237, 0, 1, 100,  21, 237, 0, 1, 100, 21]

SEQ_LENGTH=20

test_text = np.zeros((1, sentenceLength, vocabulary_size), dtype=np.bool)

for i, word in enumerate(initial_text):
  if word in dictionary:
    test_text[0, i, dictionary[word]] =1
  else:
    test_text[0, i, 0] =1

print(test_text.shape)

generated_text = []

for i in range(200):
    #why the division??
    next_word = model.predict(test_text) #(X/float(SEQ_LENGTH))

    index = sample(next_word, diversity) #np.argmax(next_character)
    generated_text.append(reverse_dictionary[index])
    
    to_insert = np.zeros((1, 1, vocabulary_size), dtype=np.bool)
    to_insert[0,0, index] = 1
    test_text= np.append(test_text, to_insert, axis=1)
    test_text= test_text[:, 1:, :]
    print(test_text.shape)


print(' '.join(generated_text))
