# Spotify Audiobook Classifier

This project is designed to create pipelines that ingest audiobooks and classify them for platforms like Spotify. Using advanced machine learning models and clustering techniques, it can dynamically group and categorize audiobooks based on their content and features. The audiobooks are being pulled from Librivox (thank you Librivox for your amazing platform), and its easy to use API for audiobook metadata, most important of which is the audiobook description.

## Features
- Audio feature extraction using Librosa
- Metadata embedding extraction with Hugging Face transformers
- KMeans clustering for unsupervised genre classification
- REST endpoints built using FastAPI, all documented within main.py

## Quickstart
Clone the repository:
```bash
git clone https://github.com/username/spotify-audiobook-classifier.git
cd spotify-audiobook-classifier
```

To run the server, use the command
```bash
uvicorn main:app --reload
```

and send API requests using any client of your choice to the main.py. Here's an example CURL request for training the model:

```bash
curl -X POST "http://127.0.0.1:8000/train_model" \
    -F "filenames=21134" \
    -F "filenames=21158" \
    -F "filenames=21040" \
    -F "filenames=21009" \
    -F "filenames=20971" \
    -F "filenames=20575" \
    -F "filenames=20457"
```

## Project Status and Future

Currently, there are issues with training the ML model. These issues are being worked on and will be fixed as soon as possible. After that, I will be testing the classifier with a validation set of new audiobooks that I will download. If all goes well, I will deploy the entire project onto AWS with dedicated pipelines that will call all relevant scripts in the proper order as follows: rename directories to the structure the code expects, extract metadata from audiobooks, and train the model (which will save its results to the models directory). Then we will also perform inference on validation sets and so on...