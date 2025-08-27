import streamlit as st
from sklearn_models import train_random_forest, predict_random_forest
from embeddings import embed_and_save_if_not
import pandas as pd

@st.cache_resource
def train(df):
    model_emb, embs = embed_and_save_if_not(df['text'].values)

    model, encoder = train_random_forest(embs, df['sentiment'], create_test=False)

    return model_emb, model, encoder

df = pd.read_csv('data/sentiment_analysis.csv')
model_emb, model, encoder = train(df)

input_data = st.text_input('Write a text to analyze')

if st.button('Predict'):
    pred = predict_random_forest(input_data, model, model_emb, encoder)
    st.write(pred[0])