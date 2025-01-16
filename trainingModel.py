from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# Preprocess the training data (Sentiment)
sentiment_texts = train_data['selected_text'].astype(str).values  # Training dataset
sentiments = train_data['sentiment'].values

# Encode the sentiment labels
sentiment_encoder = LabelEncoder()
sentiments_encoded = sentiment_encoder.fit_transform(sentiments)

# Tokenize the text data for sentiment
sentiment_tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
sentiment_tokenizer.fit_on_texts(sentiment_texts)
sentiment_sequences = sentiment_tokenizer.texts_to_sequences(sentiment_texts)

# Pad sequences to ensure uniform input length (sentiment)
max_length = 100
sentiment_padded_sequences = pad_sequences(sentiment_sequences, maxlen=max_length, padding='post', truncating='post')

# Define the model architecture
sentiment_model = Sequential([
    Embedding(input_dim=30000, output_dim=100, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(sentiments_encoded)), activation='softmax')
])

sentiment_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
sentiment_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model on the entire dataset
sentiment_model.fit(sentiment_padded_sequences, sentiments_encoded, epochs=5, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Save the trained model, tokenizer, and sentiment encoder
sentiment_model.save('sentiment_model.h5')
import pickle
with open('sentiment_tokenizer.pkl', 'wb') as file:
    pickle.dump(sentiment_tokenizer, file)
with open('sentiment_encoder.pkl', 'wb') as file:
    pickle.dump(sentiment_encoder, file)

# Testing the model with a new dataset
test_texts = test_data['text'].astype(str).values  # Separate testing dataset
test_sentiments = test_data['sentiment'].values

# Preprocess the test dataset
sentiment_test_sequences = sentiment_tokenizer.texts_to_sequences(test_texts)
sentiment_test_padded_sequences = pad_sequences(sentiment_test_sequences, maxlen=max_length, padding='post', truncating='post')
test_sentiments_encoded = sentiment_encoder.transform(test_sentiments)


# Evaluate the model on the test dataset
sentiment_loss, sentiment_accuracy = sentiment_model.evaluate(sentiment_test_padded_sequences, test_sentiments_encoded)
print(f"Sentiment Test Accuracy: {sentiment_accuracy}")

# Predict and analyze results
sentiment_predictions = sentiment_model.predict(sentiment_test_padded_sequences)
sentiment_predicted_labels = np.argmax(sentiment_predictions, axis=1)
sentiment_decoded_predictions = sentiment_encoder.inverse_transform(sentiment_predicted_labels)


# Create a DataFrame for test results
df_results = pd.DataFrame({
    'Unique_ID': test_data['textID'],
    'Original Text': test_texts,
    'Actual Sentiment': test_data['sentiment'],
    'Predicted Sentiment': sentiment_decoded_predictions,
    'sentiment_is_same': test_data['sentiment'] == sentiment_decoded_predictions,

})

df_results.to_excel("Sentiment_analysis_results.xlsx")


# Negative Multiline Comment: "I really don't understand why this app is so slow. Every time I try to open it, it freezes. The updates never seem to fix anything, and customer service is no help at all. It's incredibly frustrating and not worth the hassle."
#
# Positive One-Line Comment: "Love this app! It's super easy to use and has made my life so much more convenient."