from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import metrics
import numpy as np 


# this looks like a dataset you would download... the dataset is the truth table
# for the function whichthe neural network is trying to learn!

# xorelseweekend function:  "xor" the second two bits if first bit is 1 else "and" them
# unless its the weekend (Saturday or Sunday) then do the opposite
# 
df = np.array([
    [1,0,0,0,0,0,1], # Sunday
    [1,0,1,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,1,0,0,0,1],
    [0,1,0,0,0,0,1],
    [0,1,1,0,0,0,0],
    [1,0,0,0,0,1,0], # Monday
    [1,0,1,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,1,0,0,1,0],
    [0,0,0,0,0,1,1],
    [0,0,1,0,0,1,0],
    [0,1,0,0,0,1,0],
    [0,1,1,0,0,1,1],
    [1,0,0,0,1,0,0], # Tuesday
    [1,0,1,0,1,0,1],
    [1,1,0,0,1,0,1],
    [1,1,1,0,1,0,0],
    [0,0,0,0,1,0,1],
    [0,0,1,0,1,0,0],
    [0,1,0,0,1,0,0],
    [0,1,1,0,1,0,1],
    [1,0,0,0,1,1,0], # Wednesday
    [1,0,1,0,1,1,1],
    [1,1,0,0,1,1,1],
    [1,1,1,0,1,1,0],
    [0,0,0,0,1,1,1],
    [0,0,1,0,1,1,0],
    [0,1,0,0,1,1,0],
    [0,1,1,0,1,1,1],
    [1,0,0,1,0,0,0], # Thursday
    [1,0,1,1,0,0,1],
    [1,1,0,1,0,0,1],
    [1,1,1,1,0,0,0],
    [0,0,0,1,0,0,1],
    [0,0,1,1,0,0,0],
    [0,1,0,1,0,0,0],
    [0,1,1,1,0,0,1],
    [1,0,0,1,0,1,0], # Friday
    [1,0,1,1,0,1,1],
    [1,1,0,1,0,1,1],
    [1,1,1,1,0,1,0],
    [0,0,0,1,0,1,1],
    [0,0,1,1,0,1,0],
    [0,1,0,1,0,1,0],
    [0,1,1,1,0,1,1],
    [1,0,0,1,1,0,1], # Saturday
    [1,0,1,1,1,0,0],
    [1,1,0,1,1,0,0],
    [1,1,1,1,1,0,1],
    [0,0,0,1,1,0,0],
    [0,0,1,1,1,0,1],
    [0,1,0,1,1,0,1],
    [0,1,1,1,1,0,0],
    # [1,0,0,1,1,1,1], # there are other
    # [1,0,1,1,1,1,0], # sets of input that
    # [1,1,0,1,1,1,0], # are not valid days
    # [1,1,1,1,1,1,1],
    # [0,0,0,1,1,1,0],
    # [0,0,1,1,1,1,1],
    # [0,1,0,1,1,1,1],
    # [0,1,1,1,1,1,0],

    ])

y = df[:,-1]    # Y is the target column (last column)
X = df[:,:-1]   # X is the remaining columns

# the number of dense layers has effect on how 
# fast it converges (it converges a lot faster with 3 than 1)
model = Sequential()
for x in range(0 ,3): 
    model.add(Dense(10 , activation='relu', input_dim=6)) 
model.add(Dense(1 , activation='sigmoid'))
sgd = SGD(lr=0.1)
model.compile(
    loss='binary_crossentropy', 
    optimizer=sgd, 
    metrics=['acc'],
    )


# increased batch size (works why?)
model.fit(X, y, batch_size=10 , epochs=1000)

results = model.predict_proba(X)

for x,y in zip(X,results): 
    print ('{} -> {:.2f}'.format(x,y[0]))

