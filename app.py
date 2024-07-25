# install streamlit: pip install streamlit
# run: streamlit run app.py

import streamlit as st
import pickle
import time

# Load the model
try:
    with open('amikom_sentiment.pkl', 'rb') as file:
        model = pickle.load(file)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title('Amikom Sentiment Analysis')

tweet = st.text_input('Enter your Komen')

submit = st.button('Predict')

if submit:
    if model_loaded:
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        st.write('Prediction:', prediction[0])
    else:
        st.error("Model is not loaded. Please check the model file.")
