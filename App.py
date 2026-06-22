import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.keras')

# Helper function
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess data
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Creating prediction function

def predict_sentiment(review):
    input = preprocess_text(review)
    prediction = model.predict(input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Making app using streamlit
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    # Make prediction
    Sentiment, score = predict_sentiment(user_input)

    # Display the result
    st.write(f"Sentiment : {Sentiment}")
    st.write(f"Prediction Score : {score}")
else:
    st.write("Please enter a movie review.")
