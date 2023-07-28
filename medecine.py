import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset (replace 'your_dataset.csv' with the actual filename)
df = pd.read_excel("MEdicine_description.xlsx")

# Handle missing values (if any)
df['Description'].fillna("", inplace=True)

# Split the data into training and testing sets
X = df['Description']
y = df['Drug_Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a classification model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Function to predict the drug name for a given reason
def predict_drug(reason):
    reason_vectorized = vectorizer.transform([reason])
    prediction = model.predict(reason_vectorized)
    return prediction[0]

# Example usage:
input_reason = "head pain"
predicted_drug = predict_drug(input_reason)
print(f"Predicted Drug: {predicted_drug}")
