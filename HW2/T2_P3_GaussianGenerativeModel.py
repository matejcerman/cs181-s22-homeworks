from cmath import pi
import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.pars = ()

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        priors = [c/len(y) for c in np.unique(y, return_counts=True)[1]]
        means = []
        for c in np.unique(y):
            c_ind = np.where(y == c)
            means.append(np.mean(X[c_ind, ], axis=1))
        if self.is_shared_covariance:
            cov = np.zeros((X.shape[1], X.shape[1]))
            for i in range(len(y)):
                cov += np.outer((X[i,] - means[y[i]]), (X[i,] - means[y[i]]))
            cov = cov/len(y)
            self.pars = (np.array(priors), np.array(means), cov)
            return self.pars
        else:
            covs = []
            for c in [0,1,2]:
                cov = np.zeros((X.shape[1], X.shape[1]))
                for i in np.where(y == c)[0]:
                    cov += np.outer((X[i,] - means[y[i]]), (X[i,] - means[y[i]]))
                cov = cov/len(np.where(y == c)[0])
                covs.append(cov)
            self.pars = (np.array(priors), np.array(means), covs)
            return self.pars
            
    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        if self.is_shared_covariance:
            for x in X_pred:
                lik = [mvn.pdf(x, self.pars[1][c][0], self.pars[2]) for c in [0,1,2]]
                prior = self.pars[0]
                post = np.array([lik[c]*prior[c] for c in [0,1,2]])
                preds.append(np.argmax(post))
            return np.array(preds)
        else:
            for x in X_pred:
                lik = [mvn.pdf(x, self.pars[1][c][0], self.pars[2][c]) for c in [0,1,2]]
                prior = self.pars[0]
                post = np.array([lik[c]*prior[c] for c in [0,1,2]])
                preds.append(np.argmax(post))
            return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        preds = self.predict(X)
        if self.is_shared_covariance:
            ll = 0
            for i in range(X.shape[0]):
                ll += np.log(self.pars[0][preds[i]])
                ll += -0.5*np.log(np.linalg.det(self.pars[2])*2*pi)
                vec = X[i,] - self.pars[1][preds[i]][0]
                ll += -0.5*np.dot(np.dot(vec.T, np.linalg.inv(self.pars[2])), vec)
            return -ll
        else:
            ll = 0
            for i in range(X.shape[0]):
                ll += np.log(self.pars[0][preds[i]])
                ll += -0.5*np.log(np.linalg.det(self.pars[2][preds[i]])*2*pi)
                vec = X[i,] - self.pars[1][preds[i]][0]
                ll += -0.5*np.dot(np.dot(vec.T, np.linalg.inv(self.pars[2][preds[i]])), vec)
            return -ll
