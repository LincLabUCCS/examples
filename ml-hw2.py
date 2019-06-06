import pandas as pd
import numpy as np
import pickle
import sklearn
import sklearn.ensemble as ske
from zipfile import ZipFile

# import sklearn.cross_validation, sklearn.tree, sklearn.linear_model
# from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.svm import SVC  

from TomsKnn import TomsKNNClassifier

def main():
    data = pd.read_csv(ZipFile('./data/data.csv.zip').open('data.csv','r'), sep='|')
    X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
    y = data['legitimate'].values
    print('Researching important feature based on %i total features\n' % X.shape[1])
    # import pdb; pdb.set_trace()

    # Feature selection using Trees Classifier
    fsel = ske.ExtraTreesClassifier().fit(X, y)
    # fsel = ske.GradientBoostingClassifier(n_estimators=100).fit(X, y)

    model = SelectFromModel(fsel, prefit=True)
    X_new = model.transform(X)
    nb_features = X_new.shape[1]
    # nb_features = X.shape[1]

    # sklearn has a test_train_split (who doesn't?)    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y ,test_size=0.9)

    features = []

    print('%i features identified as important:' % nb_features)

    indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
    for f in range(nb_features):
        print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

    # XXX : take care of the feature order
    for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
        features.append(data.columns[2+f])

    myKNN = TomsKNNClassifier()

    #Algorithm comparison
    algorithms = {
            "Toms Random Classifier": TomsClassifier(),
            "Toms Constant Classifier": TomsClassifier(random=False),
            # "Toms KNN Classifier": myKNN, # takes too long to demo
            "DecisionTree": sklearn.tree.DecisionTreeClassifier(max_depth=10),
            "RandomForest": ske.RandomForestClassifier(n_estimators=100),
            "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=100),
            "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
            "Logistic Regression": LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y),
            "Gaussian Naive Bayes": GaussianNB(),
            "SVM": SVC(),
            "Perceptron":MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
            "Perceptron sgd":MLPClassifier(solver='sgd', alpha=1e-2,hidden_layer_sizes=(5, 2), random_state=1),
            "Perceptron adam":MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10,10,10,10)),
        }

    results = {}
    print("\nNow testing algorithms")
    for algo in algorithms:
        clf = algorithms[algo]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('{:>25} : {}'.format(algo, score*100))
        results[algo] = score

    winner = max(results, key=results.get)
    print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))

    # # Save the algorithm and the feature list for later predictions
    # print('Saving algorithm and feature list in classifier directory...')
    # joblib.dump(algorithms[winner], 'classifier/classifier.pkl')
    # open('classifier/features.pkl', 'wb').write(pickle.dumps(features))
    # print('Saved')

    # Identify false and true positive rates
    clf = algorithms[winner]
    res = clf.predict(X_test)
    mt = confusion_matrix(y_test, res)
    print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
    print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))


# use multiple inheritance so this class can be used
# with scikit learn ensembles
from sklearn.base import BaseEstimator, ClassifierMixin
class TomsClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,random=True):
        self.rand=random

    def fit(self, X, y):
        # I ain't learning nothing
        return self

    def predict(self, X):
        import random
        if self.rand:
            return [bool(random.getrandbits(1)) for x in X]
        else:
            return [1 for x in X]

if __name__ == '__main__':
    main()
