import os
import json
import joblib
import logging

import numpy as np

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse

from ingestion.metadata_handler import add_metadata

from processing.extract_audio_features import (
    extract_audio_features, 
    extract_features_from_directory_parallel
)
from processing.inference import perform_inference
from processing.transcribe_audiobook import transcribe_audio
from processing.extract_text_embeddings import combine_text_features_batch
from processing.audio_merger import group_audiobook_files_by_directory, concatenate_audio_files

from models.unsupervised_clustering import combine_features, perform_clustering, visualize_clusters

app = FastAPI()

# TODO: Make all endpoints flexible enough to handle both local and production servers

BASE_DIRECTORY = "./audiobooks"
METADATA_PATH = "metadata.json"
MODEL_SAVE_PATH = "./models/trained_models/kmeans_model.pkl"
MODEL_VISUALIZATION_PATH = "/models/trained_model_clusters/cluster_visualization.png"

os.makedirs(BASE_DIRECTORY, exist_ok=True)

@app.post("/upload")
async def upload_audiobook(
    file: UploadFile,
    audiobook_name: str = Form(...)
):
    """
    Uploads an audiobook file and saves it to the appropriate directory.
    
    Args:
        file (UploadFile): Uploaded audio file.
        title (str): Title of the audiobook.
        author (str): Author of the audiobook.
        audiobook_name (str): Name of the audiobook to group parts.

    Returns:
        JSONResponse: Success message with file details.
    """

    # Create a directory for the audiobook if it doesn't exist
    audiobook_dir = os.path.join(BASE_DIRECTORY, audiobook_name)
    os.makedirs(audiobook_dir, exist_ok=True)

    # Save the uploaded file to the audiobook directory
    file_path = os.path.join(audiobook_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return JSONResponse({"message": f"File {file.filename} uploaded successfully to {audiobook_name}!"})

@app.post("/process")
def process_audiobook(audiobook_name: str = Form(...), title: str = Form(...), author: str = Form(...)):
    """
    Processes all parts of the specified audiobook, merges them, and adds metadata.

    Args:
        audiobook_name (str): Name of the audiobook to process.
        title (str): Title of the audiobook for metadata.
        author (str): Author of the audiobook for metadata.

    Returns:
        JSONResponse: Success message with details of the merged file.
    """

    audiobook_dir = os.path.join(BASE_DIRECTORY, audiobook_name)
    if not os.path.exists(audiobook_dir):
        return JSONResponse({"error": f"Audiobook directory {audiobook_name} does not exist!"}, status_code=404)

    # Group and sort files in the audiobook directory
    audiobook_groups = group_audiobook_files_by_directory(BASE_DIRECTORY)

    # Ensure the specified audiobook has files to process
    if audiobook_name not in audiobook_groups or not audiobook_groups[audiobook_name]:
        return JSONResponse({"error": f"No files found for audiobook {audiobook_name}!"}, status_code=404)

    # Concatenate audio files into a single MP3
    files = audiobook_groups[audiobook_name]
    output_file = os.path.join(audiobook_dir, f"{audiobook_name}_combined.mp3")
    concatenate_audio_files(files, output_file)

    # Add metadata to the merged file
    add_metadata(output_file, title=title, author=author)

    return JSONResponse({"message": f"Audiobook {audiobook_name} processed successfully!", "file_path": output_file})

@app.get("/list_audiobooks")
def list_audiobooks():
    """
    Lists all audiobooks available in the base directory.

    Returns:
        JSONResponse: List of audiobook directories.
    """

    audiobooks = [name for name in os.listdir(BASE_DIRECTORY) if os.path.isdir(os.path.join(BASE_DIRECTORY, name))]
    return JSONResponse({"audiobooks": audiobooks})

@app.post("/train_model")
async def train_model(batch_size: int = 100):
    """
    Endpoint to train the clustering model using combined audiobook audio files and metadata.

    Args:
        filenames (list[str]): Optional list of audiobook directory names. If None, process all combined files.

    Returns:
        dict: Status message and model save path.
    """

    try:
        # Step 1: Load metadata file (needs to be generated at this point already) and collect all
        # bookid_combined.mp3 files into a list
        audio_features = {}
        audio_files = []
        all_combined_text_embeddings = []

        if not os.path.exists(METADATA_PATH):
            return {"error": "Metadata file not found. Please ensure metadata.json exists."}

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        
        metadata_dict = {entry["id"]: entry for entry in metadata}
        logging.info(f"Loaded metadata for {len(metadata_dict)} audiobooks.")

        for root, _, files in os.walk(BASE_DIRECTORY):
            for file in files:
                if file.endswith("_combined.mp3"):
                    file_id = os.path.basename(root)  # Assuming parent folder name is the audiobook ID
                    audio_path = os.path.join(root, file)
                    audio_files.append({"id": file_id, "path": audio_path})

        if not audio_files:
            return {"error": "No audio features extracted. Check your audio files."}

        logging.info(f"Discovered {len(audio_files)} combined audio files.")

        # Step 2: Extract audio features from all audiobooks within the directory
        audio_paths = [file["path"] for file in audio_files]
        audio_features_dict = extract_features_from_directory_parallel(audio_paths) 
        audio_features = {
            file["id"]: audio_features_dict[file["path"]] for file in audio_files if file["path"] in audio_features_dict
        }

        # Step 3: Loop through the files batch-wise, transcribing audio files, pulling audiobook descriptions
        # and preparing the text embeddings
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i: i + batch_size]
            logging.info(f"Processing audio batch {i // batch_size + 1}/{(len(audio_files) + batch_size - 1) // batch_size}")
            
            batch_descriptions = [file["description"] for file in batch]
            batch_transcriptions = []

            for file in batch:
                transcription = transcribe_audio(file["path"])
                batch_transcriptions.append(transcription)

            combined_text_embeddings = combine_text_features_batch(batch_descriptions, batch_transcriptions)
            all_combined_text_embeddings.extend(combined_text_embeddings)

        if not all_combined_text_embeddings:
            return {
                "error": "Error in combining audio/text embeddings extracted. Check your metadata, audio transcriptions, and audio feature extraction."
            }
        
        # Step 4: combine all text and audio embeddings
        logging.info("Combining all audio and text features for training...")
        audio_feature_matrix = np.array(list(audio_features.values()))
        text_feature_matrix = np.vstack(all_combined_text_embeddings)
        combined_features = np.hstack([audio_feature_matrix, text_feature_matrix])

        # Step 5: Train clustering model
        logging.info("Performing clustering...")
        labels = perform_clustering(combined_features)

        # Step 6: Visualize clusters
        visualize_clusters(combined_features, labels, save_path=MODEL_VISUALIZATION_PATH)
        joblib.dump({"model": labels}, MODEL_SAVE_PATH)

        return {
                "message": "Training complete", 
                "model_save_path": MODEL_SAVE_PATH,
                "visualization_path": MODEL_VISUALIZATION_PATH
        }
    
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return {"error": str(e)}
    
@app.post("/predict")
async def predict_genre(
    file: UploadFile = File(...), 
    description: str = Form(...)
):
    """
    Predicts the genre of a new audiobook by dynamically extracting audio and text features.

    Args:
        file (UploadFile): Uploaded audiobook file.
        description (str): Optional description of the audiobook.

    Returns:
        dict: Predicted genre or cluster label.
    """

    try:
        # Save the uploaded file to a temporary location
        temp_audio_path = f"temp_{file.filename}"
        with open(temp_audio_path, "wb") as f:
            f.write(file.file.read())

        # Step 1: Extract audio features
        logging.info("Extracting audio features...")
        audio_features = extract_audio_features(temp_audio_path)

        # Step 2: Transcribe the audiobook
        logging.info("Transcribing the audiobook...")
        transcription = transcribe_audio(temp_audio_path)

        # Step 3: Combine text features (transcription + description)
        combined_text_features = combine_text_features_batch([description], [transcription])[0]

        # Step 4: Combine audio and text features
        combined_features = np.hstack([audio_features, combined_text_features])

        # Step 5: Perform inference
        logging.info("Performing inference...")
        predicted_label = perform_inference(combined_features)

        # Clean up temporary audio file
        os.remove(temp_audio_path)

        return {
            "predicted_label": predicted_label,
            "description_used": description if description else "No description provided",
            "transcription_used": transcription[:200]  # Return a snippet of the transcription
        }

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}