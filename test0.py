
#SVM for classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[0:5])


X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)

X=iris.data[:,0:2]
print(X.shape)

plt.scatter(X[y==0,0],X[y==0,1],color='blue')
plt.scatter(X[y==1,0],X[y==1,1],color='red')
plt.scatter(X[y==0,1],X[y==1,0],color='cyan')
print(np.min(X),np.max(X))
plt.show()
X=iris.data[iris.target!=2,0:2]
print(X.shape)
y=iris.target[iris.target!=2]
SVMmodel = SVC(kernel='linear', C=200) # Increased C value for regularization
SVMmodel.fit(X, y)

# Evaluate the model:
accuracy = SVMmodel.score(X_test, y_test)
print(f"Accuracy of the SVM classifier with C=200: {accuracy}")


# Visualization (optional):
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Scatter plot of Iris data")
plt.show()
#supvectors = SVMmodel.support_vectors_
# Plot the support vectors here

# Separating line coefficients:
W = SVMmodel.coef_
b = SVMmodel.intercept_


#Anomaly detection via SVM
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))
plt.scatter(x[:,0], x[:,1])


SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')


scores = SVMmodelOne.score_samples(x)

thresh = quantile(scores, 0.1)
print(thresh)
index = where(scores<=thresh)
values = x[index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()
