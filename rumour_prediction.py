# NEURO-FUZZY RUMOUR PREDICTION SYSTEM (TRAINING FILE)

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DATASET (Improved)

data = {
    "text": [
        # REAL
        "Government announces new education policy",
        "Stock market rises after economic growth",
        "Scientists publish new research paper",
        "New hospital opened in city",
        "Weather department predicts rainfall",
        "University announces scholarship program",
        "City council approves development plan",
        "Health ministry releases new guidelines",
        "Tech company reports profit increase",
        "Team wins championship",

        # RUMOURS
        "BREAKING aliens landed in india share now",
        "Shocking cure for all diseases found",
        "Click here to win free iphone now",
        "Secret government plan exposed",
        "This trick will make you rich overnight",
        "Banks will shut down tomorrow withdraw money",
        "5G towers causing illness",
        "Miracle drink cures cancer instantly",
        "Government giving free money to everyone",
        "Hidden truth about moon landing revealed"
    ],
    "label": [0]*10 + [1]*10
}

df = pd.DataFrame(data)

# PREPROCESSING

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# TF-IDF

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# MODEL


model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ACCURACY

preds = model.predict(X)
print("Model Accuracy:", accuracy_score(y, preds))

# SAVE MODEL


pickle.dump(model, open("rumour_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("✅ Model saved successfully!")