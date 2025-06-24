#!/bin/bash
rm -rf /tmp/nltk_data
mkdir -p /tmp/nltk_data
python -m nltk.downloader punkt stopwords -d /tmp/nltk_data
