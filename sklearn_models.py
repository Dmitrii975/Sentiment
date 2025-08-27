from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from metrics import show_metrics
from sklearn.model_selection import train_test_split
from embeddings import get_embedding_of_text
from sklearn.preprocessing import LabelEncoder
import os
import pickle


def train_logistic(embs, y, create_test=True) -> tuple[LogisticRegression, LabelEncoder]:
    if os.path.exists('temp/encoder.pickle') and os.path.exists('temp/logistic.pickle'):

        with open('temp/encoder.pickle', 'rb') as f:
            encoder = pickle.load(f)
        with open('temp/logistic.pickle', 'rb') as f:
            lr = pickle.load(f)

        return lr, encoder
    else:
        x = embs

        encoder = LabelEncoder().fit(y)
        y = encoder.transform(y)

        lr = LogisticRegression()
        
        if create_test:
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            lr.fit(x_train, y_train)
            show_metrics(y_test, lr.predict(x_test))

            with open('temp/encoder.pickle', 'wb') as f:
                pickle.dump(encoder, f, pickle.HIGHEST_PROTOCOL)
            with open('temp/logistic.pickle', 'wb') as f:
                pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)

            return lr, encoder
        else:
            lr.fit(x, y)

            with open('temp/encoder.pickle', 'wb') as f:
                pickle.dump(encoder, f, pickle.HIGHEST_PROTOCOL)
            with open('temp/logistic.pickle', 'wb') as f:
                pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)

            return lr, encoder


def predict_logistic(text, model, embedder=None, encoder=None):
    emb = get_embedding_of_text(text, embedder)
    res = model.predict(emb.reshape((1, -1)))
    if encoder != None:
        res = encoder.inverse_transform(res)
    return res


def train_random_forest(embs, y, create_test=True):
    if os.path.exists('temp/encoder.pickle') and os.path.exists('temp/forest.pickle'):
        with open('temp/encoder.pickle', 'rb') as f:
            encoder = pickle.load(f)
        with open('temp/forest.pickle', 'rb') as f:
            lr = pickle.load(f)

        return lr, encoder
    else:
        x = embs

        encoder = LabelEncoder().fit(y)
        y = encoder.transform(y)

        lr = RandomForestClassifier()
        
        if create_test:
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            lr.fit(x_train, y_train)
            show_metrics(y_test, lr.predict(x_test))

            with open('temp/encoder.pickle', 'wb') as f:
                pickle.dump(encoder, f, pickle.HIGHEST_PROTOCOL)
            with open('temp/forest.pickle', 'wb') as f:
                pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)

            return lr, encoder
        else:
            lr.fit(x, y)

            with open('temp/encoder.pickle', 'wb') as f:
                pickle.dump(encoder, f, pickle.HIGHEST_PROTOCOL)
            with open('temp/forest.pickle', 'wb') as f:
                pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)

            return lr, encoder


def predict_random_forest(text, model, embedder=None, encoder=None):
    emb = get_embedding_of_text(text, embedder)
    res = model.predict(emb.reshape((1, -1)))
    if encoder != None:
        res = encoder.inverse_transform(res)
    return res