import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re  # Add to imports at top

# Download necessary NLTK datasets with explicit error handling
@st.cache_resource
def download_nltk_data():
    try:  # Add proper error handling
        nltk.download('punkt', quiet=True)
        nltk.download('reuters', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"NLTK download failed: {str(e)}")
        raise

@st.cache_resource(show_spinner=False)
def load_and_process_data():
    try:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
        corpus_sentences = []
        
        # Safely load Reuters data
        for fileid in reuters.fileids():
            try:
                raw_text = reuters.raw(fileid)
                
                # Modified text cleaning with regex
                tokenized_sentence = [
                    re.sub(r'[^a-z]', '', word.lower())  # Better text cleaning
                    for word in nltk.word_tokenize(raw_text) 
                    if word.isalnum() and word.lower() not in stop_words
                ]
                corpus_sentences.append(tokenized_sentence)
            except Exception as e:
                st.warning(f"Skipping file {fileid} due to error: {str(e)}")
                continue
        
        # Train Word2Vec model
        model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
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
    # Add model info sidebar
    with st.sidebar:
        st.header("Model Info")
        st.write(f"Vocabulary size: {len(model.wv)} words")
        st.write(f"Training epochs: {model.epochs}")
        st.write(f"Vector size: {model.vector_size} dimensions")
    
    # Enhanced word input section
    col1, col2 = st.columns(2)
    with col1:
        if 'current_word' not in st.session_state:
            st.session_state.current_word = "money"
        word = st.text_input("Enter a word to analyze:", 
                            value=st.session_state.current_word,
                            key="word_input")
    
    with col2:
        num_similar = st.slider("Number of similar words:", 5, 20, 10, key="num_similar")
        viz_words = st.slider("Words to visualize:", 50, 300, 100, key="viz_words")

    # Expanded analysis features
    if word.strip():
        tab1, tab2, tab3 = st.tabs(["Similar Words", "Word Relationships", "Embedding Visualization"])
        
        with tab1:
            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=num_similar)
                st.write("Top similar words:")
                st.dataframe(pd.DataFrame(similar_words, columns=['Word', 'Similarity']))
            else:
                st.warning("Word not found in vocabulary.")
        
        with tab2:
            st.write("Calculate word relationships (e.g. 'king - man + woman = queen')")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                word_a = st.text_input("Word A:", "king")
            with col_b:
                word_b = st.text_input("Subtract:", "man")
            with col_c:
                word_c = st.text_input("Add:", "woman")
            
            if st.button("Calculate Analogy"):
                try:
                    result = model.wv.most_similar(positive=[word_a, word_c], negative=[word_b])
                    st.write("Possible results:")
                    st.write(pd.DataFrame(result, columns=['Word', 'Score']))
                except KeyError as e:
                    st.error(f"Missing word in vocabulary: {e}")
        
        with tab3:
            if st.button("Generate Visualization"):
                plt.clf()
                words = list(model.wv.index_to_key)[:viz_words]
                vectors = np.array([model.wv[word] for word in words])
                tsne = TSNE(n_components=2, random_state=42)
                reduced_vectors = tsne.fit_transform(vectors)
                
                plt.figure(figsize=(12, 8))
                plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
                for i, word in enumerate(words):
                    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                               fontsize=8, alpha=0.8)
                plt.title("t-SNE Visualization of Word Embeddings")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                st.pyplot(plt)
                plt.close()
    else:
        st.warning("Please enter a word to analyze")
