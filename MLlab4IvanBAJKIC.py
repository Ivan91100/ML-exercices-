import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Defined 3 points in 2D-space:
X = np.array([[2, 1, 0], [4, 3, 0]])

# Calculate the covariance matrix:
R = np.cov(X)  # Compute the covariance matrix

# Calculate the SVD decomposition and new basis vectors:
[U, D, V] = np.linalg.svd(R)  # call SVD decomposition
u1 = U[:, 0]  # new basis vectors
u2 = U[:, 1]

# Calculate the coordinates in new orthonormal basis:
Xi1 = np.dot(u1, X)  # Projection onto first basis vector
Xi2 = np.dot(u2, X)  # Projection onto second basis vector

# Calculate the approximation of the original from new basis
X_approx = (Xi1[:, None] * u1[:, None]) + (Xi2[:, None] * u2[:, None])

# Check that you got the original
print("Original X:")
print(X)
print("Reconstructed X_approx:")
print(X_approx)

# Load Iris dataset as in the last PC lab:
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])

# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
import matplotlib.pyplot as plt
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show

# Compute pca.explained_variance_ and pca.explained_cariance_ratio_values
pca.explained_variance_
pca.explained_variance_ratio_
# Plot the principal components in 2D, mark different targets in color
plt.figure()
plt.scatter(Xpca[y == 0, 0], Xpca[y == 0, 1], color='green', label='Class 0')
plt.scatter(Xpca[y == 1, 0], Xpca[y == 1, 1], color='blue', label='Class 1')
plt.scatter(Xpca[y == 2, 0], Xpca[y == 2, 1], color='magenta', label='Class 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('2D PCA Projection')
plt.show()

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X

from sklearn.neighbors import KNeighborsClassifier
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(,)
Ypred=knn1.predict()
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(,)
ConfusionMatrixDisplay.from_predictions(,)

# Split original dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier on full 4-dimensional X
t_knn = KNeighborsClassifier(n_neighbors=3)
t_knn.fit(X_train, y_train)
y_pred_full = t_knn.predict(X_test)

# Compute and display confusion matrix for full dataset
conf_matrix_full = confusion_matrix(y_test, y_pred_full)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_full)
plt.title("Confusion Matrix - Full Dataset")
plt.show()

# Train and evaluate KNN on PCA-transformed data (first two principal components)
Xpca_2D = Xpca[:, :2]
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(Xpca_2D, y, test_size=0.2, random_state=42)
pca_knn = KNeighborsClassifier(n_neighbors=3)
pca_knn.fit(X_train_pca, y_train_pca)
y_pred_pca = pca_knn.predict(X_test_pca)

# Compute and display confusion matrix for PCA dataset
conf_matrix_pca = confusion_matrix(y_test_pca, y_pred_pca)
ConfusionMatrixDisplay.from_predictions(y_test_pca, y_pred_pca)
plt.title("Confusion Matrix - PCA (First 2 Components)")
plt.show()

# Train and evaluate KNN on original data but only using first two features
X_2D = X[:, :2]
X_train_2D, X_test_2D, y_train_2D, y_test_2D = train_test_split(X_2D, y, test_size=0.2, random_state=42)
knn_2D = KNeighborsClassifier(n_neighbors=3)
knn_2D.fit(X_train_2D, y_train_2D)
y_pred_2D = knn_2D.predict(X_test_2D)

# Compute and display confusion matrix for 2D dataset
conf_matrix_2D = confusion_matrix(y_test_2D, y_pred_2D)
ConfusionMatrixDisplay.from_predictions(y_test_2D, y_pred_2D)
plt.title("Confusion Matrix - Original X (First 2 Features)")
plt.show()
