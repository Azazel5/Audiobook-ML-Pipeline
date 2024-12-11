from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def combine_features(audio_features: dict, text_embeddings: np.array, filenames: list):
    """
    Combines audio and text features into a single feature set.

    Args:
        audio_features (dict): Dictionary mapping filenames to audio feature vectors.
        text_embeddings (np.array): Array of text embeddings.
        filenames (list): List of filenames corresponding to the text embeddings.

    Returns:
        np.array: Combined feature vectors.
    """

    combined_features = []
    for i, filename in enumerate(filenames):
        audio_vector = audio_features.get(filename, np.zeros(13))  # Default to zeros if no audio features
        text_vector = text_embeddings[i]
        combined_features.append(np.hstack([audio_vector, text_vector]))
    return np.vstack(combined_features)

def perform_clustering(features: np.array, num_clusters=5):
    """
    Performs K-means clustering on the combined feature set.

    Args:
        features (np.array): Combined feature vectors.
        num_clusters (int): Number of clusters.

    Returns:
        list: Cluster labels for each feature vector.
    """
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def visualize_clusters(features: np.array, labels: list):
    """
    Visualizes clusters using t-SNE.

    Args:
        features (np.array): Combined feature vectors.
        labels (list): Cluster labels.
    """

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE Cluster Visualization")
    plt.savefig('models/trained_models')
    plt.close()
