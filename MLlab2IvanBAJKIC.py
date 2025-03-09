import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import copy

# 0. Load dataset
loaded_points = np.load('Data/k_mean_points.npy')

plt.figure()
plt.scatter(loaded_points[:, 0], loaded_points[:, 1])
plt.title("Data Points")
plt.show()

# 1. Specify number of clusters K
k = 3


# 2. Initialize centroids
def initialize_clusters(points: np.ndarray, k_clusters: int) -> np.ndarray:
    shuffled_points = copy.deepcopy(points)
    np.random.shuffle(shuffled_points)
    centroids = shuffled_points[:k_clusters]
    return centroids


centroids = initialize_clusters(loaded_points, k)


# 3. Calculate distance from centroids to all points in datasets
def calculate_metric(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    distances = norm(points - centroid, axis=1)
    return distances


def compute_distances(points: np.ndarray, centroids_points: np.ndarray) -> np.ndarray:
    distances = np.zeros((points.shape[0], centroids_points.shape[0]))
    for idx, centroid in enumerate(centroids_points):
        distances[:, idx] = calculate_metric(points, centroid)
    return distances


# 4. Assign datapoints to the closest centroids
def assign_centroids(distances: np.ndarray) -> np.ndarray:
    assigned_centroids = np.argmin(distances, axis=1)
    return assigned_centroids


# 5. Calculate objective function
def calculate_objective(assigned_centroids: np.ndarray, distances: np.ndarray) -> float:
    objective_value = 0
    for idx, cluster_idx in enumerate(assigned_centroids):
        objective_value += distances[idx, cluster_idx]
    return objective_value


# 6. Compute new centroids
def calculate_new_centroids(points: np.ndarray, assigned_centroids: np.ndarray, k_clusters: int) -> np.ndarray:
    new_centroids = []
    for i in range(k_clusters):
        cluster_points = points[assigned_centroids == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = points[np.random.choice(range(points.shape[0]))]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


# 7. Repeat steps until convergence
def fit(points: np.ndarray, k_clusters: int, n_of_iterations: int, error: float = 0.001):
    centroids = initialize_clusters(points, k_clusters)
    prev_objective = float('inf')

    for iteration in range(n_of_iterations):
        distances = compute_distances(points, centroids)
        assigned_centroids = assign_centroids(distances)
        objective = calculate_objective(assigned_centroids, distances)

        if abs(prev_objective - objective) < error:
            print(f"Converged at iteration {iteration}")
            break

        prev_objective = objective
        centroids = calculate_new_centroids(points, assigned_centroids, k_clusters)

    return centroids, assigned_centroids


# Run K-means
final_centroids, final_assignments = fit(loaded_points, k_clusters=k, n_of_iterations=100)

# Plot results
plt.figure()
for cluster_idx in range(k):
    cluster_points = loaded_points[final_assignments == cluster_idx]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx + 1}")
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='black', marker='x', s=100, label="Centroids")
plt.legend()
plt.title("K-means Clustering Results")
plt.show()
