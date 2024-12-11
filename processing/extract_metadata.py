import requests
import json
import os

def get_metadata_by_id(audiobook_id: str):
    """
    Fetch metadata for an audiobook from the LibriVox API using its ID.

    Args:
        audiobook_id (str): The ID of the audiobook.

    Returns:
        dict: Metadata for the audiobook, or None if not found.
    """

    base_url = "https://librivox.org/api/feed/audiobooks/"
    params = {"id": audiobook_id, "format": "json"}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "books" in data and len(data["books"]) > 0:
            return data["books"][0] 
        
    return None

def fetch_metadata_for_ids(audiobook_ids: list, save_path: str = "metadata.json"):
    """
    Fetch metadata for a list of audiobook IDs and save it to a file.

    Args:
        audiobook_ids (list): List of audiobook IDs.
        save_path (str): Path to save the metadata file.
    """

    metadata_list = []

    for audiobook_id in audiobook_ids:
        print(f"Fetching metadata for ID: {audiobook_id}")
        metadata = get_metadata_by_id(audiobook_id)

        if metadata:
            metadata_list.append(metadata)
        else:
            print(f"Metadata not found for ID: {audiobook_id}")

    # Save metadata to a JSON file
    with open(save_path, "w") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Metadata saved to {save_path}")

def update_metadata_with_new_ids(new_ids: list, existing_metadata_path: str = "metadata.json"):
    """
    Updates the metadata file with new audiobook IDs.

    Args:
        new_ids (list): List of new audiobook IDs.
        existing_metadata_path (str): Path to the existing metadata file.
    """
    # Load existing metadata
    if os.path.exists(existing_metadata_path):
        with open(existing_metadata_path, "r") as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    # Fetch metadata for new IDs
    for audiobook_id in new_ids:
        print(f"Fetching metadata for new ID: {audiobook_id}")
        metadata = get_metadata_by_id(audiobook_id)
        if metadata:
            metadata_list.append(metadata)
        else:
            print(f"Metadata not found for ID: {audiobook_id}")

    # Save updated metadata
    with open(existing_metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Updated metadata saved to {existing_metadata_path}")


# audiobook_ids = ["21134", "20457", "21158", "21040", "20575", "21009", "20971"]
# fetch_metadata_for_ids(audiobook_ids)