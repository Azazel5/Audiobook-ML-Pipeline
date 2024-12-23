import torch
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel

def extract_text_embeddings(texts: list, model_name="bert-base-uncased"):
    """
    Extracts embeddings for a list of texts using a pretrained Hugging Face model.

    Args:
        texts (list): List of text data (e.g., titles or descriptions).
        model_name (str): Pretrained model to use for embedding extraction.

    Returns:
        np.array: Array of embeddings.
    """

    logging.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []

    for i, text in enumerate(texts):
        logging.info(f"Processing text {i + 1}/{len(texts)}: {text[:50]}...")

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
            embeddings.append(embedding)
        
        except Exception as e:
            logging.error(f"Error processing text {i + 1}/{len(texts)}: {e}")
            raise

    logging.info(f"Successfully processed {len(embeddings)} texts.")
    return np.vstack(embeddings)

# Combine text features for a batch of descriptions and transcriptions
def combine_text_features_batch(descriptions: list[str], transcriptions: list[str], model_name="bert-base-uncased"):
    """
    Combines text embeddings for batches of descriptions and transcriptions.

    Args:
        descriptions (list[str]): List of audiobook descriptions.
        transcriptions (list[str]): List of transcriptions for each audiobook.
        model_name (str): Pretrained model to use for embedding extraction.

    Returns:
        np.array: Combined text embedding vectors for all inputs.
    """

    if len(descriptions) != len(transcriptions):
        raise ValueError("Descriptions and transcriptions must have the same length.")

    # Extract embeddings for the batch of descriptions
    logging.info("Extracting embeddings for descriptions...")
    description_embeddings = extract_text_embeddings(descriptions, model_name)

    # Extract embeddings for the batch of transcriptions
    logging.info("Extracting embeddings for transcriptions...")
    transcription_embeddings = extract_text_embeddings(transcriptions, model_name)

    # Combine embeddings for each audiobook
    logging.info("Combining description and transcription embeddings...")
    combined_embeddings = np.hstack((description_embeddings, transcription_embeddings))  # Horizontal stack
    return combined_embeddings