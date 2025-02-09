# 💡 **TweetVibeTracker**: Your AI companion for sentiment analysis! 🔍✨

**TweetVibeTracker** is a powerful and user-friendly web application designed to analyze text sentiments using AI-driven deep learning. Whether you're processing single sentences or large datasets, TweetVibeTracker is here to help you make sense of the vibes in your data! 🚀

🔗 **Live Deployment**: [TweetVibeTracker App](https://lnkd.in/ghm7zysk)

✨ **Created by:** [Srishti Jaitly](https://www.linkedin.com/in/srishti-jaitly-6852b822b/) ✨

---

## 🎯 Features

### 1. **📥 Versatile Input Options**
- **Single Sentence Analysis**: Analyze quick sentiments of short texts.
- **Multi-Line Text Analysis**: Input paragraphs or lists for comprehensive sentiment insights.
- **Batch File Processing**: Upload Excel/CSV files for large-scale sentiment analysis, with downloadable results.

### 2. **🎨 Intuitive UI/UX**
- Dropdown menus and clear input options make navigating a breeze.
- Automatically detect file formats and preview uploads effortlessly.

### 3. **🧠 AI-Powered Sentiment Analysis**
- Built with a **TensorFlow-based sentiment model** for precise results.
- Robust preprocessing ensures accurate predictions every time.

### 4. **📊 Actionable Insights**
- Displays results instantly for single and multi-line inputs.
- Adds sentiment prediction columns to uploaded datasets for easy integration.

### 5. **🔔 Real-Time Notifications**
- Get instant toast notifications for successes and errors to stay informed.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
$ git clone https://github.com/Srish0218/TweetVibeTracker.git
$ cd tweetvibetracker
```

### 2️⃣ Install Dependencies
```bash
$ pip install -r requirements.txt
```

### 3️⃣ Place Model and Resources
Ensure the following files are in the root directory:
- `sentiment_model.h5`
- `sentiment_tokenizer.pkl`
- `sentiment_encoder.pkl`

### 4️⃣ Run the App
```bash
$ streamlit run app.py
```

---

## ✨ How to Use

### 📝 Single Sentence Input
1. Select **One-Line Sentence** from the dropdown menu.
2. Enter your sentence in the text field.
3. Click **Analyze Sentiment** to view results.

### 📄 Multi-Line Text Input
1. Choose **Multi-Line Text** from the dropdown menu.
2. Paste multiple lines of text into the text area.
3. Hit **Analyze Sentiment** to see line-by-line predictions.

### 📂 File Upload
1. Pick **Upload Excel/CSV File** from the dropdown menu.
2. Upload a file and specify the text column.
3. Process the file and download your results with predicted sentiments. 📥

---

## 🖼 Example Outputs

### 🔹 Single Sentence Input:
- Input: `"I love this product!"`
- Output: `Positive`

### 🔹 Multi-Line Input:
| Text                               | Sentiment  |
|------------------------------------|------------|
| "I love this product!"             | Positive   |
| "The service was disappointing."   | Negative   |

### 🔹 File Upload Output:
| Text                              | Sentiment  |
|-----------------------------------|------------|
| "Amazing experience!"             | Positive   |
| "Not worth the money."            | Negative   |

---

## 🛠 Technologies Used

- **Streamlit**: For a sleek and interactive user interface.
- **TensorFlow**: As the engine behind sentiment predictions.
- **Pandas**: For seamless data manipulation.
- **Pickle**: To store and load pre-trained models and tokenizers.

---

## 🚀 Future Enhancements

- 🌍 **Multi-language Support**: Expand analysis to multiple languages.
- 📈 **Confidence Scores**: Provide certainty levels for each prediction.
- 📊 **Visualization Features**: Add charts to explore sentiment trends.

---

## 🤝 Contributing

We welcome contributions! Fork the repo, create a branch, and submit your PRs to improve **TweetVibeTracker**. 💡

---

