import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
global data
# Load the trained model, tokenizer, and sentiment encoder
sentiment_model = load_model('sentiment_model.h5')  # Update the file path if necessary

with open('sentiment_tokenizer.pkl', 'rb') as file:
    sentiment_tokenizer = pickle.load(file)

with open('sentiment_encoder.pkl', 'rb') as file:
    sentiment_encoder = pickle.load(file)

# Define the maximum sequence length
MAX_SEQUENCE_LENGTH = 100

# App title and description
st.title("TweetVibeTracker")

# Dropdown for input type selection
input_type = st.selectbox(
    "Choose your input type:",
    options=["One-Line Sentence", "Multi-Line Text", "Upload Excel/CSV File"]
)

# Placeholder for user input or file upload
user_input = None
uploaded_file = None
selected_column = None

# Handle input types
if input_type == "One-Line Sentence":
    user_input = st.text_input("Enter a single sentence:", placeholder="Type a sentence here...")
elif input_type == "Multi-Line Text":
    user_input = st.text_area("Enter multi-line text:", placeholder="Type your text here...")
elif input_type == "Upload Excel/CSV File":
    uploaded_file = st.file_uploader("Upload an Excel or CSV file:", type=["csv", "xlsx"])
    if uploaded_file:
        # Display the uploaded file
        st.write("Uploaded File Preview:")
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.dataframe(data)

        # Display a dropdown to select the column for text
        column_names = data.columns.tolist()
        selected_column = st.selectbox("Select the column containing text for analysis:", column_names)

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if input_type == "One-Line Sentence" and user_input:
        # Preprocess single input
        sentiment_test_sequences = sentiment_tokenizer.texts_to_sequences([user_input])
        sentiment_test_padded_sequences = pad_sequences(
            sentiment_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
        )

        # Predict sentiment
        sentiment_predictions = sentiment_model.predict(sentiment_test_padded_sequences)
        sentiment_predicted_labels = np.argmax(sentiment_predictions, axis=1)
        sentiment_decoded_predictions = sentiment_encoder.inverse_transform(sentiment_predicted_labels)

        # Display the result
        st.success(f"**Predicted Sentiment:** {sentiment_decoded_predictions[0]}")
        # Add a Clear button
        if st.button("Clear Input"):
            st.session_state.clear()

    elif input_type == "Multi-Line Text" and user_input:
        # Split multi-line input into separate lines
        lines = user_input.split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        # Preprocess multi-line input
        sentiment_test_sequences = sentiment_tokenizer.texts_to_sequences(lines)
        sentiment_test_padded_sequences = pad_sequences(
            sentiment_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
        )

        # Predict sentiment
        sentiment_predictions = sentiment_model.predict(sentiment_test_padded_sequences)
        sentiment_predicted_labels = np.argmax(sentiment_predictions, axis=1)
        sentiment_decoded_predictions = sentiment_encoder.inverse_transform(sentiment_predicted_labels)

        # Display the results
        results = pd.DataFrame({"Text": lines, "Predicted Sentiment": sentiment_decoded_predictions})
        st.write("Multi-Line Text Sentiment Analysis Results:")
        st.dataframe(results)
        # Add a Clear button
        if st.button("Clear Input"):
            st.session_state.clear()

    elif input_type == "Upload Excel/CSV File" and uploaded_file and selected_column:
        try:
            # Preprocess the text column from the file
            texts = data[selected_column].astype(str)
            sentiment_test_sequences = sentiment_tokenizer.texts_to_sequences(texts)
            sentiment_test_padded_sequences = pad_sequences(
                sentiment_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
            )

            # Predict sentiment
            sentiment_predictions = sentiment_model.predict(sentiment_test_padded_sequences)
            sentiment_predicted_labels = np.argmax(sentiment_predictions, axis=1)
            sentiment_decoded_predictions = sentiment_encoder.inverse_transform(sentiment_predicted_labels)

            # Add predictions to the DataFrame
            data['Predicted Sentiment'] = sentiment_decoded_predictions

            # Display the input and output file previews
            st.write("Input File Preview:")
            st.dataframe(data.iloc[:, :-1])  # Display the input file columns
            st.write("Output File Preview:")
            st.dataframe(data)  # Display the file with predictions
            # Add a Clear button
            if st.button("Clear Input"):
                st.session_state.clear()

            # Provide a download option for the processed file
            st.download_button(
                label="Download Processed File",
                data=data.to_csv(index=False),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide a valid input for sentiment analysis.")


