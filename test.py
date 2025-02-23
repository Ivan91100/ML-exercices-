# =============================================================================
# SVM for Classification and Anomaly Detection Lab
#
# This lab demonstrates two applications of Support Vector Machines:
#
# 1. Classification on the Iris dataset:
#    - Load the Iris dataset and inspect its features.
#    - Perform a train/test split.
#    - Train a basic SVM classifier with a linear kernel on the full dataset.
#
#    Then, focusing on 2D binary classification:
#    - Choose only the first two features (columns) of iris.data.
#    - Since SVM in its basic form is a 2-class classifier, eliminate samples
#      where iris.target == 2.
#    - Plot scatterplots for classes 0 and 1 to check their separability.
#    - Train and test the SVM classifier (experimenting with the regularization
#      parameter C, e.g., C=200).
#    - Visualize the decision boundary using the equation: [w0, w1]*[x0, x1] + b = 0.
#    - Plot the support vectors on the 2D plot.
#
# 2. Anomaly Detection using One-Class SVM:
#    - Generate synthetic data using make_blobs.
#    - Train a One-Class SVM to detect anomalies.
#    - Plot detected anomalies using both direct prediction and score thresholding.
# =============================================================================

# ---------------------------
# PART 1: SVM Classification on Iris Data
# ---------------------------

# Import libraries for dataset handling, modeling, and visualization.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset.
iris = load_iris()
print("Feature names:", iris.feature_names)
print("First 5 samples:\n", iris.data[0:5, :])
print("First 5 target values:", iris.target[0:5])

# Use all features and all classes for initial SVM training.
X = iris.data
y = iris.target

# Split the data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape (full data):", X_train.shape)
print("Test set shape (full data):", X_test.shape)

# Train a basic SVM classifier with a linear kernel on the full dataset.
SVMmodel = SVC(kernel='linear', random_state=42)
SVMmodel.fit(X_train, y_train)
print("SVM parameters:", SVMmodel.get_params())
print("Test set accuracy (multi-class):", SVMmodel.score(X_test, y_test))

# -----------------------------------------------------------
# Now, focus on 2D binary classification for visualization.
# -----------------------------------------------------------

# Step 1: Choose only the first two features (columns) of iris.data.
X_two_features = iris.data[:, 0:2]
print("Shape with first two features:", X_two_features.shape)

# Step 2: Plot scatterplots for targets 0 and 1 to check the separability.
plt.figure(figsize=(7, 5))
plt.scatter(X_two_features[iris.target == 0, 0], X_two_features[iris.target == 0, 1],
            color='blue', label='Class 0')
plt.scatter(X_two_features[iris.target == 1, 0], X_two_features[iris.target == 1, 1],
            color='red', label='Class 1')
# (Note: The extra scatter using cyan in your code seems redundant; we focus on classes 0 & 1.)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Scatter plot for Classes 0 and 1 (First Two Features)")
plt.legend()
plt.show()

# Step 3: Eliminate samples where iris.target == 2 since SVM is a 2-class classifier.
mask = iris.target != 2
X_binary = iris.data[mask, 0:2]  # Use only the first two features.
y_binary = iris.target[mask]
print("Shape of binary data:", X_binary.shape)

# Optionally, split the binary data into train/test sets for evaluation.
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42)

# Step 4: Train the SVM classifier on binary data.
# Here, we experiment with the regularization parameter C (using C=200).
SVMmodel_bin = SVC(kernel='linear', C=200, random_state=42)
SVMmodel_bin.fit(X_train_bin, y_train_bin)

# Evaluate the binary classifier.
accuracy_bin = SVMmodel_bin.score(X_test_bin, y_test_bin)
print(f"Accuracy of the binary SVM classifier with C=200: {accuracy_bin:.2f}")

# Step 5: Visualization of the binary classifier.
# Plot the binary data points.
plt.figure(figsize=(7, 5))
plt.scatter(X_binary[y_binary == 0, 0], X_binary[y_binary == 0, 1],
            color='blue', label='Class 0')
plt.scatter(X_binary[y_binary == 1, 0], X_binary[y_binary == 1, 1],
            color='red', label='Class 1')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Binary SVM Classification (Classes 0 and 1)")

# Step 6: Plot the decision boundary.
# The decision boundary is defined by: [w0, w1] * [x0, x1] + b = 0.
# Extract the coefficients and intercept.
W = SVMmodel_bin.coef_[0]
b = SVMmodel_bin.intercept_[0]
print("Separating line coefficients (W):", W)
print("Intercept (b):", b)

# Create a range of x values for the decision boundary.
x_vals = np.linspace(np.min(X_binary[:, 0]) - 1, np.max(X_binary[:, 0]) + 1, 200)
# Compute the corresponding y values: x1 = -(w0*x0 + b) / w1.
if W[1] != 0:
    y_vals = -(W[0] * x_vals + b) / W[1]
else:
    y_vals = np.full_like(x_vals, np.mean(X_binary[:, 1]))
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

# Step 7: Plot the support vectors.
supvectors = SVMmodel_bin.support_vectors_
plt.scatter(supvectors[:, 0], supvectors[:, 1], s=100,
            facecolors='none', edgecolors='green', linewidths=1.5,
            label='Support Vectors')

plt.legend()
plt.show()

# ---------------------------
# PART 2: Anomaly Detection via One-Class SVM
# ---------------------------

from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

# Set a random seed for reproducibility.
random.seed(11)

# Generate synthetic data using make_blobs.
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, center_box=(4, 4))
plt.figure(figsize=(7, 5))
plt.scatter(x[:, 0], x[:, 1], edgecolor='k', alpha=0.7)
plt.title("Synthetic Data for Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Train a One-Class SVM.
# One-Class SVM learns the 'normal' region of data and flags points outside as anomalies.
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)

# Predict on the training data.
# The model returns 1 for inliers (normal) and -1 for outliers (anomalies).
pred = SVMmodelOne.predict(x)
anom_index = where(pred == -1)
values = x[anom_index]

# Plot the original data with anomalies highlighted in red.
plt.figure(figsize=(7, 5))
plt.scatter(x[:, 0], x[:, 1], edgecolor='k', alpha=0.7, label="Normal Data")
plt.scatter(values[:, 0], values[:, 1], color='red', label="Detected Anomalies")
plt.title("Anomaly Detection using One-Class SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.axis('equal')
plt.show()

# Alternative approach: Anomaly detection using score thresholding.
# Obtain decision function scores for each point (lower scores indicate anomalies).
scores = SVMmodelOne.score_samples(x)
# Choose a threshold at the 10th percentile.
thresh = quantile(scores, 0.1)
print("Score threshold (10th percentile):", thresh)
# Identify indices where scores are below the threshold.
index = where(scores <= thresh)
values_thresh = x[index]

# Plot anomalies based on the score threshold.
plt.figure(figsize=(7, 5))
plt.scatter(x[:, 0], x[:, 1], edgecolor='k', alpha=0.7, label="Data Points")
plt.scatter(values_thresh[:, 0], values_thresh[:, 1], color='red',
            label="Anomalies (Score Threshold)")
plt.title("Anomaly Detection via Score Thresholding")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.axis('equal')
plt.show()
