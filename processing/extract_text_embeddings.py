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
