from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
import numpy as np

EPOCHS = 500

# define documents
docs = ['atta boy',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',

        'Weak',
        'Poor effort!',
        'you did poorly',
        'poor work',
        'Could have done better.',
        ]
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
max_length = 4

vocab_size = 50  #

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)
padded_docs = pad_sequences(sequences, maxlen=max_length, padding='post')

# import pdb; pdb.set_trace()

print(padded_docs[0])
print(padded_docs[1])
print(padded_docs[2])
print(padded_docs[3])

# replaced this code with the above 
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# # pad documents to a max length of 4 words
# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length, trainable=False))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# fit the model
model.fit(padded_docs, labels, epochs=EPOCHS, verbose=0)

print('{} epochs'.format(EPOCHS))

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
print('Training Accuracy: {:.2f}'.format(accuracy))

# predict probabilities on any test set, in this
# case we just test on the training set which is 
# not best but ok for this simple example 
results = model.predict_proba(padded_docs)

for i, y in enumerate(results):
    if i==5: print('-'*23) # to separate classes
    print ('{:23s} -> {:f}'.format(docs[i],y[0]))

# repeat the encoding and padding on some test data
tests = ['good show old boy','দুনিয়া job']
codes = [one_hot(d, vocab_size) for d in tests]
padds = pad_sequences(codes, maxlen=max_length, padding='post')

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(tests)
sequences = tokenizer.texts_to_sequences(tests)
padds = pad_sequences(sequences, maxlen=max_length, padding='post')

# show results of the test set
print('-'*23)
results = model.predict_proba(padds)
for i, y in enumerate(results):
    # if i==5: print('-'*23) # to separate classes
    print ('{:23s} -> {:f}'.format(tests[i],y[0]))
