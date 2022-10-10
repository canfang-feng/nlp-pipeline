import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

stopwords = nltk.corpus.stopwords.words("english")


def join_text(text):
    """
    Join the list of words into a string separated by space, and return the result.
    """
    return " ".join(text)


def clean_text(text: str) -> str:
    """
    Clean text by removing punctuation, lowercasing, removing stopwords

    args:
        text: raw text to be cleaned
    returns:
        cleaned text
    """
    stopwords = nltk.corpus.stopwords.words("english")
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [word for word in tokens if word not in stopwords]
    return text


def tokenize(text):
    """
    Tokenizes the text.
    Args:
        text: The text to tokenize.
    Returns:
        tokens: The tokens of the text.
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")
    ]

    return clean_tokens


def calculate_word_freq(text, tokenizer):
    """
    Calculate the frequency of each word in the text, and return the result as a list of tuples.
    """
    clean_text = tokenizer(text)
    freq = {}  # stores the frequency of elements
    for x in clean_text:
        freq[x] = clean_text.count(x)

    sort_word = pd.DataFrame(freq.items(), columns=["Word", "Frequency"]).sort_values(
        by="Frequency", ascending=False
    )
    return sort_word
