from tkinter import X
import numpy as np
from scipy import stats
# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            dists = []
            for i in range(self.X.shape[0]):
                dists.append(((self.X[i,0] - x[0])/3)**2 + (self.X[i,1]- x[1])**2)
            small_ind = np.argsort(dists)
            lablist = self.y[small_ind][:self.K]
            preds.append(stats.mode(lablist, axis=None)[0])
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y