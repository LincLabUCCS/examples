from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

# xorelse function: "xor" the second two bits if first bit is 1 else "and" them
X = np.array([ [1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,0,0],[0,0,1],[0,1,0],[0,1,1] ])
y = np.array([0,1,1,0,1,0,0,1])

# this is getting hard to visualize so let's look at it as a table

# could be stored in a single array (DataFrame) like this, where 
# the last column is the y label 
# (This is the truth table for the xor-else function)
df = np.array([
    [1, 0,0,0],
    [1, 0,1,1],
    [1, 1,0,1],
    [1, 1,1,0],

    [0, 0,0,1],
    [0, 0,1,0],
    [0, 1,0,0],
    [0, 1,1,1],

    ])

X = df[:,1:3]; dim=2   # X is the columns up to the last
epochs = 500

# missing 2 binary features
df = np.array([
    [1, 1, 0,0,0],
    [1, 1, 0,1,1],
    [1, 1, 1,0,1],
    [1, 1, 1,1,0],

    [1, 0, 0,0,1],
    [1, 0, 0,1,0],
    [1, 0, 1,0,0],
    [1, 0, 1,1,1],

    [0, 1, 0,0,1],
    [0, 1, 0,1,0],
    [0, 1, 1,0,0],
    [0, 1, 1,1,1],

    [0, 0, 0,0,1],
    [0, 0, 0,1,0],
    [0, 0, 1,0,0],
    [0, 0, 1,1,1],

    ])
X = df[:,2:4]; dim=2   # X is the columns up to the last
epochs = 500


# missing 3 binary features
df = np.array([
    [1, 1, 1, 0,0,0],
    [1, 1, 1, 0,1,1],
    [1, 1, 1, 1,0,1],
    [1, 1, 1, 1,1,0],

    [1, 1, 0, 0,0,1],
    [1, 1, 0, 0,1,0],
    [1, 1, 0, 1,0,0],
    [1, 1, 0, 1,1,1],

    [1, 0, 1, 0,0,1],
    [1, 0, 1, 0,1,0],
    [1, 0, 1, 1,0,0],
    [1, 0, 1, 1,1,1],

    [1, 0, 0, 0,0,1],
    [1, 0, 0, 0,1,0],
    [1, 0, 0, 1,0,0],
    [1, 0, 0, 1,1,1],


    [0, 1, 1, 0,0,1],
    [0, 1, 1, 0,1,0],
    [0, 1, 1, 1,0,0],
    [0, 1, 1, 1,1,1],

    [0, 1, 0, 0,0,1],
    [0, 1, 0, 0,1,0],
    [0, 1, 0, 1,0,0],
    [0, 1, 0, 1,1,1],

    [0, 0, 1, 0,0,1],
    [0, 0, 1, 0,1,0],
    [0, 0, 1, 1,0,0],
    [0, 0, 1, 1,1,1],

    [0, 0, 0, 0,0,1],
    [0, 0, 0, 0,1,0],
    [0, 0, 0, 1,0,0],
    [0, 0, 0, 1,1,1],


    ])
X = df[:,3:5]; dim=2   # X is the columns up to the last
epochs = 1000

# by leaving out exactly one binary feature I get 50% probability for all tests

# leave out exactly two binary

# I might be able to guess this function by looking at the dataset
# of course it is easier if it's sorted, probably couldn't if it 
# was random order

# given the data in this format we have to split the columns
# into X array and y array for feeding to Keras
# 


X = df[:,:-1]   # X is the columns up to the last
dim = np.shape(X)[1]

# X = df[:,1:3]; dim=2   # X is the columns up to the last
y = df[:,-1]    # Y is the last or target column

#  Build the model
model = Sequential()
model.add(Dense(8, activation='tanh', input_dim=dim))
model.add(Dense(1, activation='sigmoid'))

# from keras.utils import plot_model
# plot_model( model, to_file='xorelseSimple.png', show_shapes=False, show_layer_names=False )
# plot_model( model, to_file='xorelseShapes.png', show_shapes=True,  show_layer_names=False )

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['mae', 'acc'])

model.fit(X, y, batch_size=1, epochs=epochs)

results = model.predict_proba(X)

for x,y in zip(X,results): 
    print ('{} -> {:.2f}'.format(x,y[0]))

