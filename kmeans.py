import numpy as np 
import sklearn 
from sklearn.preprocessing import scale 
from sklearn.datasets import load_digits 
from sklearn.cluster import KMeans 
from sklearn import metrics

# K-Means: unsupervised learning <- don't need training, test data 

digits = load_digits() 
data = scale(digits.data) 
y = digits.target # target variable? 

#k = len(np.unique(y)) # make it dynamic 
k = 10 
samples, features = data.shape # ex. (1000,728) 

def bench_k_means(estimator, name, data): 
    estimator.fit(data) 
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, estimator.inertia_, 
        metrics.homogeneity_score(y, estimator.labels_), # compare y values with estimators' labels <- just compare!(unsupervised) 
        metrics.completeness_score(y, estimator.labels_),
        metrics.v_measure_score(y, estimator.labels_), 
        metrics.adjusted_rand_score(y, estimator.labels_), 
        metrics.adjusted_mutual_info_score(y, estimator.labels_), 
        metrics.silhouette_score(data, estimator.labels_, metric='euclidean'))) 

clf = KMeans(n_clusters=k, init="random", n_init=10) # classifier // find how to input parameters from document! 
bench_k_means(clf, "1", data) # can see all of accuracy scores 
