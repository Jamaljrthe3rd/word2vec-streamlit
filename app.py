import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Download necessary NLTK datasets with explicit error handling
@st.cache_resource(show_spinner=False)
def download_nltk_data():
    with st.spinner('Downloading NLTK datasets...'):
        nltk.download('punkt', quiet=True)
        nltk.download('reuters', quiet=True)
        nltk.download('stopwords', quiet=True)

@st.cache_resource(show_spinner=False)
def load_and_process_data():
    try:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
        corpus_sentences = []
        
        # Add progress bar for corpus loading
        progress_bar = st.progress(0)
        fileids = reuters.fileids()
        for i, fileid in enumerate(fileids):
            try:
                raw_text = reuters.raw(fileid)
                tokenized_sentence = [
                    word.lower() 
                    for word in nltk.word_tokenize(raw_text) 
                    if word.isalnum() and word.lower() not in stop_words
                ]
                corpus_sentences.append(tokenized_sentence)
                progress_bar.progress((i+1)/len(fileids))
            except Exception as e:
                st.warning(f"Skipping file {fileid} due to error: {str(e)}")
                continue
        
        # Add model training status
        with st.spinner('Training Word2Vec model...'):
            model = Word2Vec(
                sentences=corpus_sentences, 
                vector_size=100, 
                window=5, 
                min_count=5, 
                workers=4
            )
        return model
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main app
st.title("Word2Vec Interactive Visualization")
st.write("This app trains a Word2Vec model on the Reuters dataset and allows exploration of word embeddings.")

# Load model
model = load_and_process_data()

if model is not None:
    # User input for word similarity
    word = st.text_input("Enter a word to find similar words:", "money")
    if word in model.wv:
        similar_words = model.wv.most_similar(word)
        st.write("Top similar words:")
        st.write(pd.DataFrame(similar_words, columns=['Word', 'Similarity']))
    else:
        st.write("Word not found in vocabulary.")

    # Visualization of embeddings
    if st.button("Visualize Word Embeddings"):
        words = list(model.wv.index_to_key)[:100]
        vectors = np.array([model.wv[word] for word in words])
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
        for i, word in enumerate(words):
            plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        st.pyplot(plt)
        plt.close()  # Add this to prevent memory leaks
