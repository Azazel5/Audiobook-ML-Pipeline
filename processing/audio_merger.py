import os
from pydub import AudioSegment


def group_audiobook_files_by_directory(base_directory):
    audiobook_groups = {}

    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            audiobook_groups[dir_name] = sorted(
                [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith(".mp3")]
            )

    return audiobook_groups


def concatenate_audio_files(file_list, output_path):
    combined = AudioSegment.empty()
    for file_path in file_list:
        print(f"Processing: {file_path}")
        audio = AudioSegment.from_file(file_path)
        combined += audio

    combined.export(output_path, format="mp3")
    return output_path