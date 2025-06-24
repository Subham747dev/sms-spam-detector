from model import transform_text
import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

# Load model and vectorizer
model = pickle.load(open("C:/Users/Subham/Desktop/machine learning projects/sms spam detection/model.pkl", 'rb'))
vectorizer = pickle.load(open("C:/Users/Subham/Desktop/machine learning projects/sms spam detection/vectorizer.pkl", 'rb'))


st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Check'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not a Spam Message")