from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import gensim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def build_model():
    """
    Build a pipeline that includes:
        - TfidfVectorizer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", RandomForestClassifier(n_estimators=100)),
        ]
    )
    return pipeline


# create word2vec vectors transformer
class w2vTransformer(BaseEstimator, TransformerMixin):

    """
    Wrapper class for running word2vec into pipelines and FeatureUnions
    """

    def __init__(self, w2v_model, **kwargs):
        self.model = w2v_model
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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


def build_w2v_model(w2v_model):
    """
    Build a pipeline that includes:
        - w2vTransformer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("w2v", w2vTransformer(w2v_model)),
            ("classifier", RandomForestClassifier(n_estimators=100)),
        ]
    )
    return pipeline


class d2vTransformer(BaseEstimator, TransformerMixin):

    """
    Wrapper class for running doc2vec into pipelines and FeatureUnions
    """

    def __init__(self, d2v_model, **kwargs):
        self.model = d2v_model
        self.kwargs = kwargs

    def fit(self, X, y=None):
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


def build_d2v_model(d2v_model):
    """
    Build a pipeline that includes:
        - w2vTransformer
        - RandomForestClassifier
    """
    pipeline = Pipeline(
        [
            ("w2v", d2vTransformer(d2v_model)),
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
