from sentence_transformers import SentenceTransformer
import os
import pickle


def get_embedding_of_text(texts, model=None):
    if model == None:
        model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
    embeddings = model.encode(texts)
    return embeddings


def embed_and_save_if_not(text):
    if os.path.exists('temp/tiny_bert.pickle'):
        with open('temp/tiny_bert.pickle', 'rb') as f:
            model_emb = pickle.load(f)
    else:
        model_emb = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
        with open('temp/tiny_bert.pickle', 'wb') as f:
            pickle.dump(model_emb, f, pickle.HIGHEST_PROTOCOL)

    if os.path.exists('temp/embeddings.pickle'):
        with open('temp/embeddings.pickle', 'rb') as f:
            embs = pickle.load(f)
    else:
        embs = model_emb.encode(text)
        with open('temp/embeddings.pickle', 'wb') as f:
            pickle.dump(embs, f, pickle.HIGHEST_PROTOCOL)
    return model_emb, embs
    