# 📊 Week 4 - Sentiment Analysis Model (AI & ML Internship)

## 📌 Project Overview

 The objective was to build a **Sentiment Analysis Model** capable of classifying text into **Positive** or **Negative** sentiment using Natural Language Processing (NLP) techniques.

The project demonstrates the complete machine learning pipeline, including text preprocessing, feature extraction, model training, evaluation, and prediction on custom sentences.

---

# 🎯 Objectives

* Build a sentiment analysis model using a real-world text dataset.
* Perform text preprocessing and cleaning.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression classifier.
* Evaluate model performance using Accuracy and F1-Score.
* Predict sentiment for custom user-written sentences.

---

# 📂 Dataset

**Dataset:** Twitter Sentiment Dataset

The dataset contains tweets labeled with different sentiment classes. For this project, only **Positive** and **Negative** sentiments were used to build a binary classification model.

---

# 🛠 Technologies Used

* Python
* Pandas
* Regular Expressions (re)
* Scikit-learn
* Matplotlib

---

# 📚 Machine Learning Workflow

### 1. Import Required Libraries

Imported all necessary libraries for data manipulation, text preprocessing, visualization, machine learning, and evaluation.

---

### 2. Load Dataset

Loaded the Twitter sentiment dataset and inspected its structure, columns, and data types.

---

### 3. Data Exploration

* Displayed dataset information
* Checked sentiment distribution
* Selected only Positive and Negative sentiment classes

---

### 4. Text Preprocessing

The text data was cleaned by:

* Converting text to lowercase
* Removing punctuation
* Removing numbers
* Removing special characters
* Preparing clean text for feature extraction

---

### 5. Feature Extraction

Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert cleaned text into numerical vectors that machine learning models can understand.

---

### 6. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

using a fixed random state for reproducibility.

---

### 7. Model Training

A **Logistic Regression** classifier was trained using the TF-IDF feature vectors.

---

### 8. Model Evaluation

The trained model was evaluated using:

* Accuracy Score
* F1-Score
* Confusion Matrix

These metrics measure how effectively the model classifies tweet sentiment.

---

### 9. Prediction on Custom Sentences

The trained model was tested on three custom sentences to predict whether each sentence expresses **Positive** or **Negative** sentiment.

---

### 10. Final Conclusion

The sentiment analysis model successfully classified textual data into positive and negative sentiments using Natural Language Processing techniques.

The combination of text preprocessing, TF-IDF feature extraction, and Logistic Regression produced a reliable baseline model for binary sentiment classification.

---

# 📈 Results

The model successfully:

* Cleaned and processed raw text
* Converted text into numerical features
* Learned sentiment patterns
* Classified unseen tweets
* Predicted sentiment for new user-provided sentences

---
#output
<img width="802" height="632" alt="image" src="https://github.com/user-attachments/assets/b112937a-fc87-4e39-93e9-94d63e5fa155" />

---
# ⚠ Limitations

Although the model performs well on standard text, it may struggle with:

* Sarcasm
* Irony
* Complex language
* Mixed emotions
* Previously unseen vocabulary

Future improvements could include using advanced deep learning models such as LSTM, BERT, or RoBERTa.

---

# 📁 Project Structure

```
Week4-Sentiment-Analysis/
│
├── twitter_Sentiment_Analysis.py
├── twitter_training.csv
├── confusion_matrix.png
├── README.md
└── requirements.txt
```

---

# 📌 Project Outcome

This project demonstrates practical implementation of **Natural Language Processing (NLP)** and **Machine Learning** by building an end-to-end Sentiment Analysis system capable of predicting customer opinions from textual data.

---

## 👨‍💻 Author

**Anfal Tanveer**


Week 4 - Sentiment Analysis Project
