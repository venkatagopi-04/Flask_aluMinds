#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Download the model from Google Drive
echo "Downloading rev_model.pkl from Google Drive..."
gdown https://drive.google.com/uc?id=1bYcXx0KCw8Gr0nTsTAqplDB_PZD7zEji -O rev_model.pkl

echo "Model downloaded."
