from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

# this looks like a dataset you would download... the dataset is the truth table
# for the function the neural network is trying to learn!

# xorelseweekend function:  "xor" the second two bits if first bit is 1 else "and" them
# unless its the weekend (Saturday or Sunday) then do the opposite
# 
df = np.array([
    [1,0,0,0,1], # Sunday
    [1,0,1,0,0],
    [1,1,0,0,0],
    [1,1,1,0,1],
    [0,0,0,0,0],
    [0,0,1,0,1],
    [0,1,0,0,1],
    [0,1,1,0,0],
    [1,0,0,1,0], # Monday
    [1,0,1,1,1],
    [1,1,0,1,1],
    [1,1,1,1,0],
    [0,0,0,1,1],
    [0,0,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,1],
    [1,0,0,2,0], # Tuesday
    [1,0,1,2,1],
    [1,1,0,2,1],
    [1,1,1,2,0],
    [0,0,0,2,1],
    [0,0,1,2,0],
    [0,1,0,2,0],
    [0,1,1,2,1],
    [1,0,0,3,0], # Wednesday
    [1,0,1,3,1],
    [1,1,0,3,1],
    [1,1,1,3,0],
    [0,0,0,3,1],
    [0,0,1,3,0],
    [0,1,0,3,0],
    [0,1,1,3,1],
    [1,0,0,4,0], # Thursday
    [1,0,1,4,1],
    [1,1,0,4,1],
    [1,1,1,4,0],
    [0,0,0,4,1],
    [0,0,1,4,0],
    [0,1,0,4,0],
    [0,1,1,4,1],
    [1,0,0,5,0], # Friday
    [1,0,1,5,1],
    [1,1,0,5,1],
    [1,1,1,5,0],
    [0,0,0,5,1],
    [0,0,1,5,0],
    [0,1,0,5,0],
    [0,1,1,5,1],
    [1,0,0,6,1], # Saturday
    [1,0,1,6,0],
    [1,1,0,6,0],
    [1,1,1,6,1],
    [0,0,0,6,0],
    [0,0,1,6,1],
    [0,1,0,6,1],
    [0,1,1,6,0],
    # [1,0,0,1,1,1,1], # Payday (handle it like weekend)
    # [1,0,1,1,1,1,0],
    # [1,1,0,1,1,1,0],
    # [1,1,1,1,1,1,1],
    # [0,0,0,1,1,1,0],
    # [0,0,1,1,1,1,1],
    # [0,1,0,1,1,1,1],
    # [0,1,1,1,1,1,0],

    ])

# given the data in this format we have to split the columns
# into X array and y array for feeding to Keras
# 
y = df[:,-1]    # Y is the target column (last column)
X = df[:,:-1]   # X is the remaining columns

# the number of dense layers has effect on how 
# fast it converges (it converges a lot faster with 3 than 1)
denselayers = 3
model = Sequential()
for x in range(0 ,denselayers):
    model.add(Dense(10 , activation='tanh', input_dim=4))
    # model.add(Activation('tanh')) # relu works too
model.add(Dense(1 , activation='sigmoid'))
# model.add(Activation('sigmoid'))

from keras.utils import plot_model
plot_model( model, to_file='xorelseweekendSimple.png', show_shapes=False, show_layer_names=True )
plot_model( model, to_file='xorelseweekendShapes.png', show_shapes=True,  show_layer_names=False )


sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

# increased batch size (works why?)
model.fit(X, y, batch_size=10 , epochs=10000)

results = model.predict_proba(X)

for x,y in zip(X,results): 
    print ('{} -> {:.2f}'.format(x,y[0]))
