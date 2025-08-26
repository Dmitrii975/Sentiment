from sklearn.linear_model import LogisticRegression
from metrics import show_metrics
from sklearn.model_selection import train_test_split
from embeddings import get_embedding_of_text
from sklearn.preprocessing import LabelEncoder

def train_logistic(embs, y, create_test=True) -> tuple[LogisticRegression, LabelEncoder]:
    x = embs

    encoder = LabelEncoder().fit(y)
    y = encoder.transform(y)

    lr = LogisticRegression()
    
    if create_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        lr.fit(x_train, y_train)
        show_metrics(y_test, lr.predict(x_test))
        return lr, encoder
    else:
        lr.fit(x, y)
        return lr, encoder


def predict_logistic(text, model, embedder=None, encoder=None):
    emb = get_embedding_of_text(text, embedder)
    res = model.predict(emb.reshape((1, -1)))
    if encoder != None:
        res = encoder.inverse_transform(res)
    return res