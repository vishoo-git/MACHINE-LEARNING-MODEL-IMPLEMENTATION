# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY -- CODTECH IT SOLUTIONS

NAME : Vishok N.M

INTERN ID : :CT08UZY

DOMAIN : Python Programming

BATCH DURATION: FEB 15TH,2025 TO MAR 15TH,2025

MENTOR NAME: NEELA SANTHOSH

DESCRIPTION:
Project: Spam Email Detection Using Machine Learning
Objective:
The goal of this project is to build a predictive model that can classify SMS messages as either spam or ham (non-spam) using machine learning techniques. We'll use the SMS Spam Collection Dataset which contains labeled SMS messages, and we'll apply natural language processing (NLP) methods and machine learning algorithms to classify the messages.

Steps:
Data Loading and Exploration:

First, we load the SMS Spam Collection Dataset, which consists of two columns: label (either "ham" or "spam") and message (the SMS content).
We perform basic exploration to understand the structure of the data and preprocess it for machine learning.
Data Preprocessing:

The labels in the dataset are converted into numeric values (0 for "ham" and 1 for "spam").
Text preprocessing includes converting the text messages into numerical representations using techniques like CountVectorizer, which converts text into a bag-of-words model. This allows the model to understand the frequency of words and use them as features for classification.
Stop words (commonly used words like "the", "is", etc.) are removed to reduce noise in the data.
Train-Test Split:

The dataset is split into training and testing sets. 80% of the data is used to train the model, while 20% is reserved for testing and evaluating the model’s performance.
Model Training:

The model is trained using a Multinomial Naive Bayes classifier, which is well-suited for text classification problems. Naive Bayes is a probabilistic classifier that works well with word frequencies (count data), making it ideal for spam detection tasks.
Evaluation:

After training, the model is evaluated using the accuracy score, classification report, and confusion matrix.
The accuracy score gives us the overall percentage of correct predictions.
The classification report includes key metrics like precision, recall, and F1-score, which help assess how well the model performs for both classes (ham and spam).
The confusion matrix is used to visualize the true positives, false positives, true negatives, and false negatives, helping us understand the model's error patterns.
Visualization:

A heatmap of the confusion matrix is generated using Seaborn to help visualize how many messages were correctly or incorrectly classified as spam/ham.
Expected Outcomes:
Model Performance:
The model typically achieves an accuracy of 98-99%,

OUTPUT
