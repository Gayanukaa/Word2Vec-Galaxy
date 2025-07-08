import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import time
from pathlib import Path

class WordVectorAnalyzer:
    def __init__(self, model_name='word2vec-google-news-300'):
        """Initialize with pre-trained word vectors"""
        self.model = self._load_model_with_progress(model_name)
        self.vocab = set(self.model.key_to_index.keys())


    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.vocab:
            return self.model[word]
        return None

    def word_analogy(self, word1, word2, word3):
        """
        Compute: word3 - word1 + word2
        Example: biggest - big + small = smallest
        """
        try:
            # Vector arithmetic: X = vector(word3) - vector(word1) + vector(word2)
            result_vector = (self.model[word3] - self.model[word1] + self.model[word2])

            # Find most similar word to result vector
            similar_words = self.model.similar_by_vector(result_vector, topn=5)
            return similar_words[0][0]  # Return the most similar word
        except KeyError as e:
            return f"Word not found: {e}"

    def find_similar_words(self, word, num_words=20):
        """Find similar words using cosine similarity"""
        if word not in self.vocab:
            return []

        similar_words = self.model.most_similar(word, topn=num_words)
        return [word] + [w[0] for w in similar_words]

    def reduce_dimensions(self, words, method='pca', n_components=3):
        """Reduce word vectors to 3D for visualization"""
        vectors = np.array([self.model[word] for word in words if word in self.vocab])
        valid_words = [word for word in words if word in self.vocab]

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)

        reduced_vectors = reducer.fit_transform(vectors)
        return valid_words, reduced_vectors