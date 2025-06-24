import streamlit as st
import pickle
from model import transform_text

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    st.subheader("Prediction:")
    st.success("Spam" if result == 1 else "Not Spam")
