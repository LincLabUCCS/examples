from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
import sys

# this function is simple you could probably
# guess the function by looking at the data
X = np.array([[0],[1],[2],[3]])
y = np.array([ 0,  1,  1,  0] )

model = Sequential()
model.add(Dense(18, activation='tanh', input_dim=1))
model.add(Dense(1, activation='sigmoid'))

from keras.utils import plot_model
plot_model( model, to_file='xorSimple.png', show_shapes=False, show_layer_names=False )
plot_model( model, to_file='xorShapes.png', show_shapes=True,  show_layer_names=False )


sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, epochs=2000)
results = model.predict_proba(X)

for x,y in zip(X,results):
    print ('{} -> {}'.format(x,y))

"""
[[ 0.0033028 ]
 [ 0.99581173]
 [ 0.99530098]
 [ 0.00564186]]
"""