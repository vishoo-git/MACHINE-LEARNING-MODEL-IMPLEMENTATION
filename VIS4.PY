import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
file = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Step 2: Data Preprocessing
# Checking the first few records in the dataset
print(file.head())

# Convert labels ('ham' and 'spam') to numerical values
file['label'] = file['label'].map({'ham': 0, 'spam': 1})

# Step 3: Text Vectorization
# We will use the CountVectorizer to convert text into a bag-of-words model
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(file['message'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, file['label'], test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
