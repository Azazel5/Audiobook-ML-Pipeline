import logging
import numpy as np
from sklearn.externals import joblib

def perform_inference(features: np.array):
    """
    Predicts the cluster/genre for a given feature vector.

    Args:
        features (np.array): Combined audio and text features.

    Returns:
        str: Predicted genre or cluster label.
    """

    try:
        # Load the trained clustering model
        model_path = "./models/trained_models/kmeans_model.pkl" 
        clustering_model = joblib.load(model_path)

        # Perform prediction
        cluster = clustering_model.predict([features])[0]
        return f"Cluster {cluster}" 

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise
