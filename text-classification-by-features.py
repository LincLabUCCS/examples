import re
import os
import sys
import numpy as np

import nltk
from nltk.corpus import stopwords

from keras.utils import to_categorical

from sklearn.preprocessing import normalize

import sklearn.ensemble as ske
import sklearn.tree, sklearn.linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from os.path import join

# Simple hard-coded state-machine to count syllables
def syllables(str):
    state = 'consonant'
    syllables = 0
    for char in str:
        if char in 'aeiouy':
          if state == 'consonant':
            state = 'vowel_1'
            syllables += 1
          elif state == 'vowel_1':
            state = 'vowel_2'
          elif state == 'vowel_2':
            state = 'consonant'
            syllables += 1
        else:
          state = 'consonant'
    # fix silent e, English is weird
    if ( syllables > 1 and str[-1] is 'e' and  str[-2] != 'l'):
      syllables -= 1
    return syllables;

# turn text into a set of features... whatever you like
def text2features(text,normalize=True):
  stop_words = set(stopwords.words('english'))
  # [w for w in word_tokens if not w in stop_words] 

  words             = [w for w in re.split("[^a-zA-Z0-9]", text) if not w.strip() in stop_words ]
  char_count        = len(text)
  word_count        = len(words)
  # proper_count      = len(proper_words)
  max_length        = np.max([len(x) for x in words])
  max_syllables     = np.max([syllables(x) for x in words])
  average_length    = np.mean([len(x) for x in words])
  average_syllables = np.mean([syllables(x) for x in words])

  features = np.array([
    char_count, 
    word_count, 
    # proper_count, 
    max_length, 
    max_syllables, 
    average_length, 
    average_syllables,
    ])
  
  if normalize:
    features = [f/np.linalg.norm(features) for f in features]

  return features

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
    X.append(text2features(text))
    y.append(label)

  return X,y

def main():

  X,y = get20newsgroups()

  # Let sklearn split the data into train and test 
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

  # Algorithm comparison
  models = {
          "DecisionTree": sklearn.tree.DecisionTreeClassifier(max_depth=10),
          "RandomForest": ske.RandomForestClassifier(n_estimators=100),
          "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=100),
          "AdaBoost": ske.AdaBoostClassifier(n_estimators=1000),
          "Logistic Regression": LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train),
          "Gaussian Naive Bayes": GaussianNB(),
          "SVM": SVC(gamma='auto'),
          "Perceptron":MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
          "Perceptron sgd":MLPClassifier(solver='sgd', alpha=1e-2,hidden_layer_sizes=(5, 2), random_state=1),
          "Perceptron adam":MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10,10,10,10)),
      }

  results = {}
  print("Testing models...")
  for name,clf in models.items():
      # clf = models[model]
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      print('{:>25} : {}'.format(name, score*100))
      results[name] = score

  winner = max(results, key=results.get)
  print('Winning model is %s with a %f %% success' % (winner, results[winner]*100))

if __name__ == '__main__':
    main()
