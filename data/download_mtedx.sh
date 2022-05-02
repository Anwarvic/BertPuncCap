#!/usr/bin/env bash

# install gdown
echo "Installing prerequisites..."
pip install gdown

# download file from GoogleDrive
echo "Downloading data (15.7MB)..."
gdown --id 1yQZ1Sjb1SOOtjWtfrio92VWTlx00l6-9

# decompress the downloaded file
echo "Extracting the data..."
unzip mTEDx2021_transcription.zip -d mTEDx