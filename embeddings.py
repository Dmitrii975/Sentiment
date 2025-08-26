from sentence_transformers import SentenceTransformer

def get_embedding_of_text(texts, model=None):
    if model == None:
        model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
    embeddings = model.encode(texts)
    return embeddings