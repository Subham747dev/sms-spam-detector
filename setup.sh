#!/bin/bash

# Create target directory
mkdir -p /tmp/nltk_data

# Download required nltk data non-interactively
python -m nltk.downloader punkt stopwords -d /tmp/nltk_data
