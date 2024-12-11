import os
import json
import joblib

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from processing.audio_merger import group_audiobook_files_by_directory, concatenate_audio_files
from processing.extract_audio_features import extract_audio_features, extract_features_from_directory
from processing.extract_text_embeddings import extract_text_embeddings
from ingestion.metadata_handler import add_metadata
from models.unsupervised_clustering import combine_features, perform_clustering, visualize_clusters

app = FastAPI()

# TODO: Make all endpoints flexible enough to handle both local and production servers

BASE_DIRECTORY = "./audiobooks"
METADATA_PATH = "metadata.json"
model_save_path = "./models/trained_models/kmeans_model.pkl"

audio_dir = "./audiobooks"
model_save_path = "./models/kmeans_model.pkl"
metadata_path = "./metadata.json"

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
async def train_model(filenames: list[str] = Form(None)):
    """
    Endpoint to train the clustering model using combined audiobook audio files and metadata.

    Args:
        filenames (list[str]): Optional list of audiobook directory names. If None, process all combined files.

    Returns:
        dict: Status message and model save path.
    """
    try:
        # Step 1: Load metadata
        if not os.path.exists(metadata_path):
            return {"error": "Metadata file not found. Please ensure metadata.json exists."}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Step 2: Determine which files to process
        if filenames:
            matched_metadata = [entry for entry in metadata if entry["id"] in filenames]
            if not matched_metadata:
                return {"error": "No matching metadata found for the provided filenames."}

            descriptions = [entry["description"] for entry in matched_metadata]
            combined_audio_files = [
                os.path.join(audio_dir, filename, f"{filename}_combined.mp3") for filename in filenames
            ]
            for file in combined_audio_files:
                if not os.path.exists(file):
                    return {"error": f"Audio file not found: {file}"}

            audio_features = {
                filename: extract_audio_features(file)
                for filename, file in zip(filenames, combined_audio_files)
            }
        else:
            # Process all '_combined.mp3' files dynamically
            audio_features = extract_features_from_directory(audio_dir)
            descriptions = [
                entry["description"] for entry in metadata if f"{entry['id']}_combined.mp3" in audio_features
            ]

        # Step 3: Extract text embeddings
        text_embeddings = extract_text_embeddings(descriptions)

        # Step 4: Combine features
        combined_features = combine_features(audio_features, text_embeddings, list(audio_features.keys()))

        # Step 5: Perform clustering
        labels = perform_clustering(combined_features, num_clusters=5)

        visualization_path = "/models/trained_model_clusters/cluster_visualization.png"
        visualize_clusters(combined_features, labels, save_path=visualization_path)

        # Step 6: Save clustering model
        joblib.dump({"model": labels}, model_save_path)
        return {
            "message": "Training complete", 
            "model_save_path": model_save_path,
            "visualization_path": visualization_path
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_genre")
async def predict_genre(filenames: list[str] = Form(None)):
    """
    Endpoint to predict the genre cluster for combined audiobook files.

    Args:
        filenames (list[str]): Optional list of audiobook directory names. If None, process all combined files.

    Returns:
        dict: Predicted cluster labels.
    """
    try:
        # Step 1: Load the saved clustering model
        if not os.path.exists(model_save_path):
            return {"error": "Model not trained yet. Please call /train_model first."}

        saved_model = joblib.load(model_save_path)
        kmeans_model = saved_model["model"]

        # Step 2: Load metadata
        if not os.path.exists(metadata_path):
            return {"error": "Metadata file not found. Please ensure metadata.json exists."}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Step 3: Determine which files to process
        if filenames:
            matched_metadata = [entry for entry in metadata if entry["id"] in filenames]
            if not matched_metadata:
                return {"error": "No matching metadata found for the provided filenames."}

            descriptions = [entry["description"] for entry in matched_metadata]
            combined_audio_files = [
                os.path.join(audio_dir, filename, f"{filename}_combined.mp3") for filename in filenames
            ]
            for file in combined_audio_files:
                if not os.path.exists(file):
                    return {"error": f"Audio file not found: {file}"}

            audio_features = {
                filename: extract_audio_features(file)
                for filename, file in zip(filenames, combined_audio_files)
            }
        else:
            # Process all '_combined.mp3' files dynamically
            audio_features = extract_features_from_directory(audio_dir)
            descriptions = [
                entry["description"] for entry in metadata if f"{entry['id']}_combined.mp3" in audio_features
            ]

        # Step 4: Extract text embeddings
        text_embeddings = extract_text_embeddings(descriptions)

        # Step 5: Combine features
        combined_features = combine_features(audio_features, text_embeddings, list(audio_features.keys()))

        # Step 6: Predict cluster labels
        cluster_labels = kmeans_model.predict(combined_features)
        return {"predicted_clusters": {filename: label for filename, label in zip(audio_features.keys(), cluster_labels)}}

    except Exception as e:
        return {"error": str(e)}
