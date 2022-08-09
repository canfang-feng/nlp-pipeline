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
    df["text"], df["label"], test_size=0.2, random_state=42
)

print("Building model...")

# w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
# model = build_w2v_model(w2v_model)

# use doc2vec model
tagged_docs = [
    gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)
]
d2v_model = gensim.models.Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=2)
model = build_d2v_model(d2v_model)

print("Fitting model...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Evaluating model...")
evaluate_model(y_test, y_pred)
