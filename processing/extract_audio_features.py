import os
import psutil
import librosa
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        logging.info(f"Memory usage before processing {file_path}: {mem_before:.2f} MB")
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mem_after = process.memory_info().rss / (1024 * 1024)
        logging.info(f"Memory usage after processing {file_path}: {mem_after:.2f} MB")
        logging.info(f"Extracted features for {file_path}")
        return np.mean(mfcc, axis=1)
    
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        raise

def extract_features_from_directory(files: list[str]):
    """
    Extract audio features in batches for a list of files.

    Args:
        files (list[str]): List of audio file paths.

    Returns:
        dict: Dictionary mapping filenames to feature vectors.
    """

    features = {}
    for file_path in files:
        try:
            features[os.path.basename(file_path)] = extract_audio_features(file_path)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    return features

def parallel_audio_extraction(files: list[str]):
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