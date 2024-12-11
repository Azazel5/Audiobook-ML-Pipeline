import librosa
import numpy as np

def extract_audio_features(file_path: str):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_var": np.var(mfcc, axis=1).tolist(),
    }
    return features