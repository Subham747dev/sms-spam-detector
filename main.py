

import nltk

# 🚫 Clear all previous paths to avoid loading corrupted punkt_tab or wrong data
nltk.data.path.clear()

# ✅ Set only the correct path
nltk.data.path.append("/tmp/nltk_data")

# ✅ Download required resources to the correct path
nltk.download('punkt', download_dir="/tmp/nltk_data")
nltk.download('stopwords', download_dir="/tmp/nltk_data")

# ❌ Remove this line – it's unnecessary and may load local junk:
# nltk.data.path.append('./nltk_data')



import streamlit as st
from model import transform_text
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("SMS Spam Detector")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
