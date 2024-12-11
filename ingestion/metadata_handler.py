import pandas as pd

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1

def add_metadata(file_path, title, author):
    audio = MP3(file_path, ID3=ID3)

    audio["TIT2"] = TIT2(encoding=3, text=title) 
    audio["TPE1"] = TPE1(encoding=3, text=author)
    audio.save()

def preprocess_metadata(metadata: dict):
    metadata_df = pd.DataFrame([metadata])
    metadata_df.fillna("Unknown", inplace=True)
    metadata_df.columns = metadata_df.columns.str.lower()
    return metadata_df