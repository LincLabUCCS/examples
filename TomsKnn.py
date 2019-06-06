import numpy as np
import sklearn
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# use multiple inheritance so this class can be used
# with scikit learn ensembles etc.
from sklearn.base import BaseEstimator, ClassifierMixin
class TomsKNNClassifier(BaseEstimator, ClassifierMixin):

    """
    Whatever parameters your classifier takes are initialized
    in this python constructor
    """
    def __init__(self,K=5,ord=5):
        self.K=K
        self.ord=ord

    """
    KNN doesn't really do any "training" it just uses the 
    training set data during the prediction phase, so just 
    store the data in the object during "fit" or "training"
    """
    def fit(self, X, y):
        self.X=X
        self.y=y
        return self

    """
    Given a testing set... return predictions
    """
    def predict(self, X):
        # return an array of labels for each feature set in X
        return np.asarray([self._predict(self.X,b,self.y,self.K) for b in X])

    """
    Given a set of training vectors A (from training) and a
    single feature vector b (from testing), return the label 
    that best classifies b; get labels from Y (training labels)
    """
    def _predict(self,A,b,Y,k):
        # calculate distances between vector b and each vector a in A
        # get the corresponding labels (from Y) of nearest K neighbors
        # return the label that has the most "votes" from the K neighbors

        #  hints: 
        # you can use np.linalg.norm to calculate distance between two vectors
        # you can use collections.Counter to select the label with maximum values
        # in a list of labels (votes)

        # I'll let you write this code yourself.
        # We can look at my solution after the homework is collected
        distances = [ np.linalg.norm(a-b,self.ord) for a in A ]
        votes = Y[np.argsort(distances)[0:k]]

        # use counter to count votes
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]

    def bestParameters(self,X,y):
        k_range = list(range(1, 6))
        ord_range = list([2,3,10])
        parameters = dict(K=k_range, ord=ord_range)
        print('searching for best parameters: {} ...'.format(parameters))

        clf = GridSearchCV(self, parameters, cv=5,iid=False, n_jobs=-1, pre_dispatch=2)
        clf.fit(X, y)
        return clf.best_params_

def main():
    print("\nTesting KNN classifier for HW 2...\n")

    # key: parameter name
    # value: list of values that should be searched for that parameter
    # single key-value pair for param_grid
    k_range = list(range(1, 31))
    ord_range = list(range(1,5))
    # k_range = list(range(1, 31))




    import numpy as np
    from sklearn import datasets
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target


    # Split iris data in train and test data
    # A random permutation, to split the data randomly
    # NOTE: this is an example of train_test_split() that so
    # many libraries offer (this allocates 10 to testing)
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-20]]
    iris_y_train = iris_y[indices[:-20]]
    iris_X_test = iris_X[indices[-20:]]
    iris_y_test = iris_y[indices[-20:]]


    knn = TomsKNNClassifier()
    print("best parameters (whole set):",knn.bestParamters(iris_X,iris_y))
    print("best parameters (train set):",knn.bestParamters(iris_X_train,iris_y_train))
    print("best parameters ( test set):",knn.bestParamters(iris_X_test,iris_y_test))

    # clf = GridSearchCV(knn, parameters)
    # clf.fit(iris_X_test, iris_y_test)
    # print(clf.best_params_)

    exit()


    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(cv=5)
    knn.fit(iris_X_train, iris_y_train) 

    y_pred = knn.predict(iris_X_test)
    y_true = iris_y_test

    print("sklearn's KNN", y_pred)
    print("sklearn's KNN", y_true)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    print(accuracy)

    # Create and fit a nearest-neighbor classifier
    knn = TomsKNNClassifier(K=5)
    knn.fit(iris_X_train, iris_y_train) 

    y_pred = knn.predict(iris_X_test)
    y_true = iris_y_test

    print("    Tom's KNN", y_pred)
    print("    Tom's KNN", y_true)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    print(accuracy)

if __name__ == '__main__':
    main()
