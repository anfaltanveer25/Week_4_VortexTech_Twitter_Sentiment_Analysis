"""
Week 4 AI & ML Internship
Sentiment Analysis using Twitter Dataset

This script performs the following major tasks:
1. Loads the raw Twitter sentiment dataset
2. Inspects the distribution of sentiment classes and filters to
   only the two classes we want to classify (Positive / Negative)
3. Cleans the raw tweet text (lowercasing, removing noise)
4. Converts the cleaned text into numerical features using TF-IDF
5. Splits data into training and testing sets
6. Trains a Logistic Regression classifier on the TF-IDF features
7. Evaluates the model using Accuracy, F1-score, and a Confusion Matrix
8. Tests the trained model on new, unseen custom sentences
9. Summarizes the overall approach and its limitations
"""

# ==========================================================
# STEP 1: Import Required Libraries
# Description:
# Import libraries for data handling, text preprocessing,
# machine learning, evaluation, and visualization.
#
# - pandas: loading and manipulating the tabular dataset
# - re: regular expressions used for cleaning raw text
# - matplotlib.pyplot: plotting the confusion matrix
# - train_test_split: splitting data into training/testing sets
# - TfidfVectorizer: converting text into numerical (TF-IDF) features
# - LogisticRegression: the classification model used for sentiment
# - accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay:
#   tools to evaluate how well the model performs
# ==========================================================

import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================================
# STEP 2: Load Dataset
# Description:
# Load the Twitter sentiment dataset from a CSV file.
# The file has no header row, so column names are assigned manually:
# "id" (tweet identifier), "entity" (the brand/topic the tweet is about),
# "label" (the sentiment class), and "text" (the actual tweet content).
# df.head() previews the first 5 rows, and df.info() shows column
# data types and non-null counts to spot any structural issues early.
# ==========================================================
df = pd.read_csv("twitter_training.csv", header=None)
df.columns=["id","entity","label","text"]
print(df.head())
print(df.info())

# ==========================================================
# STEP 3: Check Class Distribution
# Description:
# Display how many tweets fall into each sentiment label
# (e.g. Positive, Negative, Neutral, Irrelevant). This helps
# understand class balance before modeling.
#
# Since this project focuses on a binary Positive vs Negative
# classifier, all other classes (Neutral, Irrelevant) are dropped
# here, keeping only the rows labeled "Positive" or "Negative".
# ==========================================================
print(df["label"].value_counts())

# Keep Positive and Negative only
df=df[df["label"].isin(["Positive","Negative"])].copy()

# ==========================================================
# STEP 4: Clean Text
# Description:
# Raw tweets contain a lot of noise (mixed casing, punctuation,
# numbers, hashtags, special characters) that add little value
# to a text classifier and can even hurt performance. The
# clean_text() function:
#   1. Converts all text to lowercase, so words like "Great" and
#      "great" are treated as the same token.
#   2. Uses a regular expression to strip out anything that is
#      not a letter or whitespace (numbers, punctuation, emojis,
#      symbols, etc.), leaving only clean alphabetic words.
# The cleaned version of each tweet is stored in a new column,
# "cleaned_text", which will be used for feature extraction.
# ==========================================================
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    return text

df["cleaned_text"]=df["text"].apply(clean_text)

# ==========================================================
# STEP 5: TF-IDF Feature Extraction
# Description:
# Machine learning models cannot work directly with raw text,
# so it must be converted into numerical features. TF-IDF
# (Term Frequency - Inverse Document Frequency) measures how
# important a word is to a specific tweet relative to the entire
# dataset: common words across all tweets get a lower weight,
# while distinctive words that strongly characterize a tweet get
# a higher weight.
#
# max_features=5000 limits the vocabulary to the 5000 most
# informative words, keeping the feature space manageable and
# reducing the risk of overfitting on rare words.
#
# X holds the resulting TF-IDF feature matrix (input features),
# and y holds the corresponding sentiment labels (target variable).
# ==========================================================
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df["cleaned_text"])
y=df["label"]

# ==========================================================
# STEP 6: Train-Test Split
# Description:
# Split the TF-IDF features (X) and labels (y) into a training
# set (80%) and a testing set (20%). The model learns patterns
# only from the training set, and its performance is then
# validated on the unseen testing set to estimate how well it
# will generalize to new, real-world tweets.
# random_state=42 makes the split reproducible across runs.
# ==========================================================
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# ==========================================================
# STEP 7: Train Logistic Regression
# Description:
# Logistic Regression is a simple yet effective linear model
# commonly used for binary text classification tasks. It is
# trained (fit) on the training TF-IDF features and their
# corresponding Positive/Negative labels.
# max_iter=1000 increases the number of optimization iterations
# allowed for the solver to converge, since TF-IDF feature spaces
# can be large and may need more iterations than the default.
# ==========================================================
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# ==========================================================
# STEP 8: Evaluate Model
# Description:
# Use the trained model to predict sentiment labels for the
# unseen test set, then measure how well those predictions match
# the true labels using two key metrics:
#   - Accuracy: the overall percentage of correctly classified tweets.
#   - F1 Score: the harmonic mean of precision and recall for the
#     "Positive" class, giving a balanced measure of performance
#     especially useful when class distribution is not perfectly even.
#
# A Confusion Matrix is also generated and plotted, showing the
# counts of correct and incorrect predictions for each class
# (True Positive, False Positive, True Negative, False Negative).
# This visualization is saved as an image file for reporting.
# ==========================================================
pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred))
print("F1 Score:",f1_score(y_test,pred,pos_label="Positive"))

cm=confusion_matrix(y_test,pred,labels=["Positive","Negative"])
ConfusionMatrixDisplay(cm,display_labels=["Positive","Negative"]).plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================================
# STEP 9: Test Custom Sentences
# Description:
# To sanity-check the model beyond the test set, a few hand-written
# example sentences are passed through the same cleaning function
# and the same TF-IDF vectorizer (transform, not fit_transform,
# since the vocabulary is already learned from training data).
# The trained model then predicts a sentiment label for each
# custom sentence, and the results are printed side by side with
# the original sentence for easy inspection.
# ==========================================================
examples=[
"This was the best experience ever.",
"I really hate this product.",
"The service was amazing and I would recommend it."
]
X_new=vectorizer.transform([clean_text(x) for x in examples])
preds=model.predict(X_new)
for s,p in zip(examples,preds):
    print(f"{s} -> {p}")

# ==========================================================
# STEP 10: Final Summary
# Description:
# This project cleaned raw text, converted it into TF-IDF
# features, trained a Logistic Regression classifier,
# evaluated it using Accuracy and F1-score, and predicted
# sentiment for custom sentences.
#
# Limitation:
# The model may struggle with sarcasm, irony, or unseen words.
# ==========================================================
