import os
import librosa
import logging
import numpy as np


def extract_audio_features(file_path: str):
    """
    Extracts audio features (e.g., MFCCs) from an MP3 file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.array: Mean MFCC features.
    """

    logging.info(f"Extracting audio features from {file_path}...")

    try:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        logging.info(f"Extracted features for {file_path}")
        return np.mean(mfcc, axis=1)
    
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        raise

def extract_features_from_directory(directory: str):
    """
    Extracts audio features for all '_combined.mp3' files in subdirectories.

    Args:
        directory (str): Path to the main directory containing subdirectories with combined audio files.

    Returns:
        dict: Dictionary mapping filenames to feature vectors.
    """
    features = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_combined.mp3"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing file {file_path}...")
                try:
                    features[file] = extract_audio_features(file_path)
                except Exception as e:
                    logging.error(f"Skipping file {file_path} due to error: {e}")
    return features