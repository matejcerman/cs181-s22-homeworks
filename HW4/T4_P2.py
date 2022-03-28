# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import cluster
import seaborn as sns

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        K = self.K
        N = X.shape[0]
        L = []

        # initialize responsibility vectors randomly
        r = np.zeros((N,K))
        r[np.array(range(r.shape[0])),np.random.randint(0, K, size=(N,1)).flatten()] = 1
        
        # compute cluster centers
        mu = np.array([np.mean([X[j,] for j in range(N) if r[j,k]==1], axis = 0) for k in range(K)])

        # main loop
        for i in range(10):
            L.append(0)

            # make cluster assignments while computing loss
            for n in range(N):
                Ln = ((X[n,]-mu)**2).sum(axis = 1)
                L[-1] += Ln[np.argmax(r[n,])]
                r[n,] = np.array([j == np.argmin(Ln) for j in range(K)]).astype(int)
            
            # compute cluster centers
            mu = np.array([np.mean([X[j,] for j in range(N) if r[j,k]==1], axis = 0) for k in range(K)])
        
        self.r = r
        self.mu = mu
        self.L = L

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
    
    def fit(self, X):
        self.X = X
        clusters = []
        clusters.append([[n] for n in range(X.shape[0])])
        cldists = []

        for t in range(X.shape[0]-1):
            no = len(clusters[-1])
            cldists.append(np.zeros((no,no)))
            for p in range(no):
                for q in range(no): 
                    D = cdist(X[clusters[-1][p]], X[clusters[-1][q]])
                    if self.linkage == "max":
                        cldists[-1][p,q] = np.max(D)
                    elif self.linkage == "min":
                        cldists[-1][p,q] = np.min(D)
                    else:
                        cldists[-1][p,q] = np.sum((X[clusters[-1][p]].mean(axis = 0) -X[clusters[-1][q]].mean(axis = 0))**2)
            x,y = np.where(cldists[-1] == np.min(cldists[-1][np.triu_indices(cldists[-1].shape[0], k=1)]))
            first = min(x[0], y[0])
            second = max(x[0],y[0])
            merged = clusters[-1][first] + clusters[-1][second]
            copy = clusters[-1][:]
            del copy[first]
            del copy[second-1]
            clusters.append(copy + [merged])
        
        self.clusters = clusters

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        cl = self.clusters[-n_clusters]
        means = [self.X[c,].mean(axis=0) for c in cl]
        return np.array(means)

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    if standardized:
        plt.savefig("KMeans_centroids standardized.png")
    else:
        plt.savefig("KMeans_centroids.png")
    plt.close()

KMeans1 = KMeans(10)
KMeans1.fit(large_dataset)
plt.figure(figsize=(10,10))
plt.plot(range(1,11), KMeans1.L)
plt.title("KMeans Loss as a Function of Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss (objective function")
plt.savefig("kmeans_loss.png")
plt.show()
plt.close()

# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)
# ~~ Part 3 ~~
meanv = large_dataset.mean(axis = 0)
sdv = large_dataset.std(axis=0)
large_dataset_standardized = (large_dataset-meanv)/np.array([s if s != 0 else 1 for s in sdv])
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()
plt.close()

# TODO: Write plotting code for part 5
sols = [HAC(l) for l in LINKAGES]
for i in range(len(sols)):
    sols[i].fit(small_dataset)
    counts = [len(c) for c in sols[i].clusters[-n_clusters]]
    plt.figure(figsize=(10,10))
    plt.bar(x = range(1,n_clusters+1), height = counts)
    plt.title("Number of points in clusters for HAC with " + LINKAGES[i] +" linkages")
    plt.xlabel("Cluster index")
    plt.ylabel("Number of images in cluster")
    plt.savefig("HAC cluster numbers " + LINKAGES[i] +".png")
    plt.close()

km = KMeans(K=n_clusters)
km.fit(large_dataset)
counts = km.r.sum(axis=0)
plt.figure(figsize=(10,10))
plt.bar(x = range(1,n_clusters+1), height = counts)
plt.title("Number of points in clusters for KMeans")
plt.xlabel("Cluster index")
plt.ylabel("Number of images in cluster")
plt.savefig("KMeans cluster numbers.png")
plt.close()

# TODO: Write plotting code for part 6
kms = KMeans(K=n_clusters)
kms.fit(small_dataset)
kmscl = [h[0].tolist() for h in [np.where(np.argmax(kms.r, axis = 1)==i) for i in range(n_clusters)]]

cls = [sols[0].clusters[-n_clusters], sols[1].clusters[-n_clusters], sols[2].clusters[-n_clusters], kmscl]

confs = []
for i1 in range(len(cls)):
    for i2 in range(i1, len(cls)):
        if cls[i1] != cls[i2]:
            confs.append(np.zeros((n_clusters, n_clusters)))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    for m in cls[i1][i]:
                        if m in cls[i2][j]:
                            confs[-1][i][j] += 1
names = ["HAC max and HAC min", "HAC max and HAC centroid", "HAC max and KMeans", "HAC min and HAC centroid", "HAC min and KMeans", "HAC centroid and KMeans"]
for i in range(len(confs)):
    ax = sns.heatmap(confs[i])
    ax.set_title("Confusion heatmap of " + names[i])
    fig = ax.get_figure()    
    fig.savefig("heatmap " + str(i) +".png")
    plt.close()