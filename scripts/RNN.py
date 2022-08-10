# Load the cleaned training and test sets
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import pandas as pd
import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from clean_text import clean_text
from sklearn.model_selection import train_test_split


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def build_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def tokenize_pad_sequences(tokenizer, texts, maxlen=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)


print("Loading data...")
df = pd.read_csv("../data/spam.csv", encoding="latin-1")
df["clean_text"] = df["text"].apply(lambda x: clean_text(x))

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# Train the tokenizer and use that tokenizer to convert the sentences to sequences of numbers
tokenizer = build_tokenizer(X_train)
X_train_seq_padded = tokenize_pad_sequences(tokenizer, X_train, maxlen=50)
X_test_seq_padded = tokenize_pad_sequences(tokenizer, X_test, maxlen=50)

# Construct our basic RNN model framework
model = Sequential()
model.add(Embedding(len(tokenizer.index_word) + 1, 32))
model.add(LSTM(32, dropout=0, recurrent_dropout=0))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", precision_m, recall_m],
)

# Fit the RNN
history = model.fit(
    X_train_seq_padded,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test_seq_padded, y_test),
)
