from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scripts.util import clean_text
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
from sklearn.metrics import confusion_matrix


def build_model():
    """
    Build a pipeline that includes:
        - TfidfVectorizer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(tokenizer=clean_text)),
            ("classifier", RandomForestClassifier(n_estimators=100)),
        ]
    )
    return pipeline


# create word2vec vectors transformer
class w2vTransformer(BaseEstimator, TransformerMixin):

    """
    Wrapper class for running word2vec into pipelines and FeatureUnions
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        # persist model for later use
        w2v_model = gensim.models.Word2Vec(X, vector_size=100, window=5, min_count=2)
        self.model = w2v_model
        return self

    def transform(self, X):
        X = X.copy()
        words = set(self.model.wv.index_to_key)
        X_vect = np.array(
            [np.array([self.model.wv[i] for i in ls if i in words]) for ls in X],
            dtype=object,
        )
        X_vect_avg = []
        for i in X_vect:
            if len(i) > 0:
                X_vect_avg.append(np.mean(i, axis=0))
            else:
                X_vect_avg.append(np.zeros(100))

        return np.array(X_vect_avg)

    def get_feature_names(self):
        return self.model.wv.index2word

    def get_params(self, deep=True):
        return {"model": self.model}


def build_w2v_model():
    """
    Build a pipeline that includes:
        - w2vTransformer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("w2v", w2vTransformer()),
            ("classifier", RandomForestClassifier(n_estimators=100)),
        ]
    )
    return pipeline


class d2vTransformer(BaseEstimator, TransformerMixin):

    """
    Wrapper class for running doc2vec into pipelines and FeatureUnions
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        # persist model for later use
        tagged_docs = [
            gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X)
        ]
        d2v_model = gensim.models.Doc2Vec(
            tagged_docs, vector_size=100, window=5, min_count=2
        )
        self.model = d2v_model

        return self

    def transform(self, X):
        tagged_docs_X = [
            gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X)
        ]
        X_vect = [self.model.infer_vector(eval(str(v.words))) for v in tagged_docs_X]

        return X_vect

    def get_feature_names(self):
        return self.model.docvecs.index2entity

    def get_params(self, deep=True):
        return {"model": self.model}


def build_d2v_model():
    """
    Build a pipeline that includes:
        - w2vTransformer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("w2v", d2vTransformer()),
            ("classifier", RandomForestClassifier(n_estimators=100)),
        ]
    )
    return pipeline


def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance
    """
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(
        "Precision: {} / Recall: {} / Accuracy: {}".format(
            round(precision, 3),
            round(recall, 3),
            round((y_pred == y_test).sum() / len(y_pred), 3),
        )
    )


def plot_confusion_matrix(cm: np.array):
    fig, ax = plt.subplots(1, figsize=(4, 4))
    sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    plt.show()
