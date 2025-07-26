import os
import time
from pathlib import Path

import gensim.downloader as api
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA


class WordVectorAnalyzer:
    """Handles Word2Vec model loading and word vector operations"""

    def __init__(self, model_name="word2vec-google-news-300"):
        """Load Word2Vec model and prepare vocabulary"""
        self.model = self._load_model_with_progress(model_name)
        self.vocab = set(self.model.key_to_index.keys())

    def _load_model_with_progress(self, model_name):
        """Download and load Word2Vec model with progress indicator"""
        model_path = Path(api.base_dir) / model_name

        if not os.path.exists(model_path):
            st.info("� Downloading Word2Vec model (1.5GB)...")
            progress_bar = st.progress(0)

            # Simple progress simulation since gensim doesn't provide callbacks
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.05)

            model = api.load(model_name)
            progress_bar.empty()
            st.success("✅ Model downloaded successfully!")
        else:
            st.info("📂 Loading cached model...")
            model = api.load(model_name)
            st.success("✅ Model loaded from cache!")

        return model

    def get_vector(self, word):
        """Get vector representation of a word"""
        return self.model[word] if word in self.vocab else None

    def word_analogy(self, word1, word2, word3):
        """Calculate word analogy: word3 - word1 + word2 = result"""
        missing_words = [w for w in [word1, word2, word3] if w not in self.vocab]
        if missing_words:
            return f"Words not found: {', '.join(missing_words)}"

        try:
            result_vector = self.model[word3] - self.model[word1] + self.model[word2]
            similar_words = self.model.similar_by_vector(result_vector, topn=20)

            input_words = {word1.lower(), word2.lower(), word3.lower()}

            for word, similarity in similar_words:
                if word.lower() not in input_words:
                    return word

            return "No valid analogy found"
        except Exception as e:
            return f"Error: {str(e)}"

    def find_similar_words(self, word, num_words=20):
        """Find words similar to the given word"""
        if word not in self.vocab:
            return []

        similar_words = self.model.most_similar(word, topn=num_words)
        return [word] + [w[0] for w in similar_words]

    def reduce_dimensions(self, words):
        """Convert high-dimensional word vectors to 3D coordinates using PCA"""
        vectors = np.array([self.model[word] for word in words if word in self.vocab])
        valid_words = [word for word in words if word in self.vocab]

        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(vectors)
        return valid_words, reduced_vectors
