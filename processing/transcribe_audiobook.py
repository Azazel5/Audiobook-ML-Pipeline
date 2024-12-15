import whisper

def transcribe_audio(file_path: str):
    """
    Transcribes spoken content from an audio file using Whisper.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Transcription of the audio content.
    """
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]