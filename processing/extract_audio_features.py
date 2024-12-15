import os
import psutil
import librosa
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def extract_audio_features(file_path: str):
    """
    Extracts enhanced audio features (MFCC, Chroma, ZCR, etc.) from an MP3 file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: Dictionary of extracted features.
    """

    logging.info(f"Extracting audio features from {file_path}...")
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000, duration=30)  # Limit to 30 seconds for efficiency

        # Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=y), axis=1)

        # Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_mean = np.mean(harmonic)
        percussive_mean = np.mean(percussive)

        # Aggregate features into a dictionary
        features = {
            "mfcc": mfcc.tolist(),
            "chroma": chroma.tolist(),
            "zcr": zcr.tolist(),
            "spectral_contrast": spectral_contrast.tolist(),
            "rms": rms.tolist(),
            "harmonic_mean": harmonic_mean,
            "percussive_mean": percussive_mean,
        }

        logging.info(f"Extracted features for {file_path}")
        return features

    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        raise

def extract_features_from_directory_parallel(files: list[str]):
    """
    Extract audio features in parallel for a list of files.

    Args:
        files (list[str]): List of file paths.

    Returns:
        dict: Dictionary of features for each file.
    """

    features = {}

    def process_file(file_path):
        try:
            logging.info(f"Extracting audio features from {file_path}...")
            return file_path, extract_audio_features(file_path)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return file_path, None

    with ThreadPoolExecutor(max_workers=9) as executor:
        results = executor.map(process_file, files)

    for file_path, feature in results:
        if feature is not None:
            features[file_path] = feature

    return features