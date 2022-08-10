from pyexpat import model
import pandas as pd
from clean_text import clean_text
from sklearn.model_selection import train_test_split
from engineer_feature import *

import gensim
from sklearn.pipeline import Pipeline

print("Loading data...")
df = pd.read_csv("../data/spam.csv", encoding="latin-1")
df["clean_text"] = df["text"].apply(lambda x: clean_text(x))

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)
print(X_train[:1])
print("Building model...")


# model = build_w2v_model()

model = build_d2v_model()

# model = build_model()

print("Fitting model...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Evaluating model...")
evaluate_model(y_test, y_pred)
