import sklearn
from sklearn import datasets
from sklearn import svm 
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier 

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train) 

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2) # check the list of kernels on document! // (C=1)=>(C=2) <- affects accuracy 
#clf = svm.SVC(kernel="poly", degree=2) # polynomial <- takes time! 
clf.fit(x_train, y_train) 

y_pred = clf.predict(x_test) 

acc = metrics.accuracy_score(y_test, y_pred) 

print(acc) # by changing kernels, the accuracy can be improved! 



# cf. K-Neighbors Classifier 
clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train) 

y_pred = clf.predict(x_test) 

acc = metrics.accuracy_score(y_test, y_pred) 

print(acc) # SVM is generally better! 
