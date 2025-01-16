from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
sentiment_model = load_model('sentiment_model.h5')  # Update the file path if necessary

# Save the trained model, tokenizer, and sentiment encoder
with open('sentiment_tokenizer.pkl', 'rb') as file:
    sentiment_tokenizer = pickle.load(file)
with open('sentiment_encoder.pkl', 'rb') as file:
    sentiment_encoder = pickle.load(file)


# Take input for sentiment analysis
test_text = input("Enter Tweet: ")

# Preprocess the input text
sentiment_test_sequences = sentiment_tokenizer.texts_to_sequences([test_text])  # Wrap in a list to ensure it's processed correctly
sentiment_test_padded_sequences = pad_sequences(sentiment_test_sequences, maxlen=100, padding='post', truncating='post')

# Predict sentiment
sentiment_predictions = sentiment_model.predict(sentiment_test_padded_sequences)
sentiment_predicted_labels = np.argmax(sentiment_predictions, axis=1)
sentiment_decoded_predictions = sentiment_encoder.inverse_transform(sentiment_predicted_labels)

# Output the predicted sentiment
print('Predicted Sentiment:', sentiment_decoded_predictions[0])
