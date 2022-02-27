import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        
        X = np.array([[1] + x for x in X])
        
        W = np.random.rand(X.shape[1], 3)
        truey = np.zeros((len(y), 3))
        for i in range(len(y)):
            truey[i, y[i]] = 1
        self.y = truey

        self.yh = []

        for r in range(self.runs):
            yhat = np.zeros((len(y), 3))
            for i in range(len(y)):
                Wxi = np.dot(W.T, X[i,])
                yhat[i,] = np.exp(Wxi - special.logsumexp(Wxi))
            self.yh.append(yhat)

            grad = np.zeros((3, X.shape[1]))
            for j in [0,1,2]:
                grad[j] = np.dot((yhat.T[j]- truey.T[j]).T, X)
            grad = grad.T
            
            W = W - self.eta * grad + 2*self.lam*W
        self.we = W       

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        #X_pred = np.array([[1] + x for x in X_pred])
        for x in X_pred:
            Wx = np.dot((self.we).T, x)
            yhat = np.exp(Wx - special.logsumexp(Wx))
            preds.append(np.argmax(yhat))
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        
        yax = []
        for r in range(self.runs):
            yax.append(0)
            for v in range(self.y.shape[0]):
                for c in range(self.y.shape[1]):
                    if self.y[v,c] == 0 or self.yh[r][v,c]<0.0000000001:
                        yax[r] += 0
                    else:
                        lyhat = np.log(self.yh[r][v,c])
                        yax[r] += -lyhat

        xax = range(self.runs)

        plt.figure()
        plt.plot(xax, yax)
        plt.title("Loss over iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Negative Log-Likelihood Loss")
        plt.savefig(output_file + ".png")

        if show_charts:
            plt.show()
        plt.close()
