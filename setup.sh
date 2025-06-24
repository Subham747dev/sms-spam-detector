#!/bin/bash

# Download required nltk data to a writable folder
mkdir -p /tmp/nltk_data

python -m nltk.downloader -d /tmp/nltk_data punkt stopwords