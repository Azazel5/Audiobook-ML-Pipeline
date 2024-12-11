from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from processing.audio_merger import group_audiobook_files_by_directory, concatenate_audio_files
from ingestion.metadata_handler import add_metadata, preprocess_metadata

app = FastAPI()


# TODO: Make all endpoints flexible enough to handle both local and production servers


# Local storage for uploaded files
BASE_DIRECTORY = "./audiobooks"
os.makedirs(BASE_DIRECTORY, exist_ok=True)

@app.post("/upload")
async def upload_audiobook(
    file: UploadFile,
    title: str = Form(...),
    author: str = Form(...),
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
