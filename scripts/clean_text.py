import nltk
import re
import string

stopwords = nltk.corpus.stopwords.words("english")


def clean_text(text: str) -> str:
    """
    Clean text by removing punctuation, lowercasing, removing stopwords

    args:
        text: raw text to be cleaned
    returns:
        cleaned text
    """
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [word for word in tokens if word not in stopwords]
    return text


# usage example: messages["clean_text"] = messages["text"].apply(lambda x: clean_text(x))
