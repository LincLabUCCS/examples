import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from os.path import join

# ######################################################################
# Retrieve the GloVe embeddings if not already downloaded
# utility function (in utils.py) to download file and cache it
from utils import getFileUrl
from zipfile import ZipFile

# the location to store downloaded data files, if you don't have the 
# file cached locally it will take a long time, the first time
DATA_DIR = './data'
getFileUrl( 'http://nlp.stanford.edu/data/glove.6B.zip', # url source
            DATA_DIR)                                    # where to cache
#########################################################################

GLOVE_ZIP = join(DATA_DIR,'glove.6B.zip')
GLOVE_TXT = 'glove.6B.300d.txt' # or 'glove.6B.100d.txt' for example

BASE_DIR = ''
TEXT_DATA_DIR = join(DATA_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

EPOCHS = 2

print('Indexing word vectors...')

embeddings_index = {}

# leave the glove file zipped and open the text file within
with ZipFile(GLOVE_ZIP) as myzip:
    with myzip.open(GLOVE_TXT) as f:
      for line in f:
          values = line.split()
          word = values[0]
          coefs = np.asarray(values[1:], dtype='float32')
          embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# retrieve 20 newsgroups as X y data, download if needed
def get20newsgroups(data_home='./data'):
  from sklearn.datasets import fetch_20newsgroups
  print('Processing 20 newsgroup dataset...')
  texts = fetch_20newsgroups(data_home=data_home, 
    subset='all', 
    # categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'], 
    shuffle=True, random_state=42,
    remove=('headers', 'footers')
    )
  X = []
  y = []
  for label,text in zip(texts.target,texts.data):
    X.append(text)
    y.append(label)

  return X,y

texts,labels = get20newsgroups()

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS);
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


# print (texts[0][0:10])

# import pdb
# pdb.set_trace()

print('Found {} unique tokens.'.format(len(word_index)))

# before to_categorical() the labels are numbers 0-19 
# for 20 newsgroups
# np.asarray(labels)
# array([ 0,  0,  0, ..., 19, 19, 19])

labels = to_categorical(np.asarray(labels))

# after to_categorical they are one-hot vectors
# of length 20
# to_categorical(np.asarray(labels))
# array([[1., 0., 0., ..., 0., 0., 0.],
#        [1., 0., 0., ..., 0., 0., 0.],
#        [1., 0., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 1.],
#        [0., 0., 0., ..., 0., 0., 1.],
#        [0., 0., 0., ..., 0., 0., 1.]], dtype=float32)

print('Shape of data tensor:', data.shape) # (7532, 1000)
print('Shape of label tensor:', labels.shape) # (7532, 20)

# Let sklearn split the data into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( data, labels, test_size=0.33, random_state=42)

print('Preparing embedding matrix.')
# how many words, from the vocabulary, are we using?
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

# initialize embedding matrix to all zeros
# Out of Vocabulary (OOV) Words will remain all-zeros.
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_NUM_WORDS: continue
    embedding_vector = embeddings_index.get(word)

    # update vector for words in the vocabulary 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print (np.shape(embedding_matrix))
print('Build Keras Functional Model...')
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print(np.shape(sequence_input))
# exit()

# create the embedding layer 
embedded_sequences = Embedding(num_words,
                            EMBEDDING_DIM,
                            # embeddings_initializer=Constant(embedding_matrix),
                            # weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True,
                            name='helloWorld'
                            )(sequence_input)
# setting weights explicitly and trainable to true gives
# accuracy:[2.994886415961857, 0.04585679809316357]

# with default initializer and default trainable (true)
# accuracy:[2.5902353896312147, 0.14078841517265056]
# accuracy:[2.537627508059108, 0.13153660501190856]

# using pre-trained with trainable=False 
# accuracy:[2.994915429182045, 0.04585679809316357]

# embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(20, activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=128,
          epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT,
          )

# results = model.predict(x_test)
# for i, y in enumerate(results):
#     if i==5: print('-'*23) # to separate classes
#     print ('{} -> {}'.format(labels[i],y[0]))


accuracy = model.evaluate(x_test, y_test)
print ('accuracy:{}'.format(accuracy))
