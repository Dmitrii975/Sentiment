import streamlit as st
from sklearn_models import train_logistic, predict_logistic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


st.title('Sentiment detection')

@st.cache_resource
def train(df):
    model_emb = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
    embs = model_emb.encode(df['text'].values)

    model, encoder = train_logistic(embs, df['sentiment'], create_test=False)

    return model_emb, model, encoder

df = pd.read_csv('data/sentiment_analysis.csv')
model_emb, model, encoder = train(df)

input_data = st.text_input('Write a text to analyze')

if st.button('Predict'):
    pred = predict_logistic(input_data, model, model_emb, encoder)
    st.write(pred[0])
