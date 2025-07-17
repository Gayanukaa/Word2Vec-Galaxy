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

    def _load_model_with_progress(self, model_name):
        """Load model with progress bar"""
        # Check if model is already downloaded
        model_path = Path(api.base_dir) / model_name

        if not os.path.exists(model_path):
            # Model needs to be downloaded
            st.info(f"🔄 Downloading {model_name} model... This may take a few minutes.")
            st.write("📊 Model size: ~1.5GB")

            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Show animated progress while downloading
            status_text.text("🌐 Connecting to download server...")

            # Since gensim doesn't provide progress callbacks, we'll simulate progress
            # based on typical download time
            import threading
            import time

            download_complete = False

            def download_model():
                nonlocal download_complete
                try:
                    model = api.load(model_name)
                    download_complete = True
                    return model
                except Exception as e:
                    st.error(f"Error downloading model: {e}")
                    download_complete = True
                    return None

            # Start download in a separate thread
            model_result = [None]

            def download_thread():
                model_result[0] = download_model()

            thread = threading.Thread(target=download_thread)
            thread.start()

            # Show progress animation
            progress = 0
            messages = [
                "📡 Downloading model files...",
                "🔄 Processing word vectors...",
                "📦 Extracting model data...",
                "⚡ Optimizing for fast access...",
                "✅ Finalizing setup..."
            ]

            message_index = 0
            while not download_complete:
                # Update progress bar
                progress = min(progress + 1, 95)  # Don't go to 100% until actually done
                progress_bar.progress(progress)

                # Update status message
                if progress % 20 == 0 and message_index < len(messages) - 1:
                    message_index += 1

                status_text.text(messages[message_index])
                time.sleep(0.1)

            # Wait for thread to complete
            thread.join()

            # Complete the progress bar
            progress_bar.progress(100)
            status_text.text("✅ Download complete!")

            # Clear the progress indicators after a short delay
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            if model_result[0] is not None:
                st.success("🎉 Model loaded successfully!")
                return model_result[0]
            else:
                st.error("❌ Failed to load model")
                return None

        else:
            # Model is already downloaded, just load it
            st.info("📂 Loading cached model...")
            with st.spinner("Loading model from cache..."):
                model = api.load(model_name)
            st.success("✅ Cached model loaded successfully!")

        return model

    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.vocab:
            return self.model[word]
        return None

    def word_analogy(self, word1, word2, word3):
        """
        Compute: word3 - word1 + word2
        Example: king - man + woman = queen

        Note: Excludes input words from results to avoid the common issue where
        Word2Vec returns one of the input words due to high similarity in embedding space.
        """
        try:
            # Check if all words exist in vocabulary
            missing_words = []
            for word in [word1, word2, word3]:
                if word not in self.vocab:
                    missing_words.append(word)

            if missing_words:
                return f"Word not found: {', '.join(missing_words)}"

            # Vector arithmetic: result = word3 - word1 + word2
            result_vector = self.model[word3] - self.model[word1] + self.model[word2]

            # Find most similar words, excluding the input words
            # Use a larger topn to ensure we find non-input words
            similar_words = self.model.similar_by_vector(result_vector, topn=20)

            # Create comprehensive set of input words to exclude (case-insensitive)
            input_words_lower = {word1.lower(), word2.lower(), word3.lower()}
            input_words_exact = {word1, word2, word3}

            # Filter out the input words and their variations
            for word, similarity in similar_words:
                if (word.lower() not in input_words_lower and
                    word not in input_words_exact):
                    return word

            # If somehow all results are input words (very rare), return the best match
            # with a warning message
            if similar_words:
                return f"{similar_words[0][0]} (Warning: May be input word)"
            else:
                return "No similar words found"

        except Exception as e:
            return f"Error in analogy calculation: {str(e)}"

    def find_similar_words(self, word, num_words=20):
        """Find similar words using cosine similarity"""
        if word not in self.vocab:
            return []

        try:
            similar_words = self.model.most_similar(word, topn=num_words)
            return [word] + [w[0] for w in similar_words]
        except Exception as e:
            st.error(f"Error finding similar words: {e}")
            return []

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