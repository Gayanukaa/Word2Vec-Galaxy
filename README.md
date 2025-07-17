# Word2Vec-Galaxy 🌌

An interactive 3D visualization tool for exploring high-dimensional word vectors and performing vector arithmetic operations. Built with Streamlit and Plotly, this application transforms complex word embeddings into intuitive 3D visualizations, making semantic relationships and vector operations visually comprehensible.

## 🚀 Features

- **3D Word Vector Visualization** - Explore semantically similar words in interactive 3D space using PCA dimensionality reduction
- **Vector Arithmetic Operations** - Perform and visualize classic word analogies (king - man + woman = queen) with step-by-step vector operations
- **Interactive Controls** - Intuitive sidebar controls for customizing visualizations and selecting parameters
- **Progress Bar Downloads** - Smart model downloading with animated progress indicators for better user experience
- **Dark Mode Interface** - Modern dark theme optimized for data visualization
- **Real-time Processing** - Instant visualization updates with loading indicators and error handling

## 📦 Repository Structure

```
Word2Vec-Galaxy/
┣ .streamlit/
┃ ┗ config.toml                   ← Dark mode configuration
┣ main.py                         ← Streamlit frontend application
┣ word_vectors.py                 ← Word2Vec model handler with progress loading
┣ visualization.py                ← 3D plotting functions for words and analogies
┣ examples.ipynb                  ← Jupyter notebook with usage examples
┣ requirements.txt                ← Python dependencies
┗ README.md                       ← Project documentation
```

## 🛠️ Getting Started

Follow these steps to set up and run the Word2Vec Galaxy visualization tool locally.

### 1. Prerequisites

• **Python 3.11+** (Conda or Miniconda recommended)
• **Internet connection** (for downloading Word2Vec model on first run)
• **4GB+ RAM** (for handling the Word2Vec model)

### 2. Clone the Repository

```bash
git clone https://github.com/Gayanukaa/Word2Vec-Galaxy.git
cd Word2Vec-Galaxy
```

### 3. Create & Activate Conda Environment

```bash
conda create -n word2vec python=3.11 -y
conda activate word2vec
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

From the project root:

```bash
streamlit run main.py
```

The app will open in your browser (usually at `http://localhost:8501`).

### First Run Setup

- The application will automatically download the Google News Word2Vec model (~1.5GB) on first launch
- A progress bar will show download status and estimated completion time
- Subsequent runs will use the cached model for faster startup

## 🎯 How to Use

### Similar Words Visualization

1. Enter a word in the sidebar (e.g., "pet", "computer", "happiness")
2. Adjust the number of similar words using the slider (5-50)
3. Click "🔍 Visualize Similar Words"
4. Explore the 3D plot where colors indicate semantic similarity

### Vector Arithmetic

1. Enter three words for the analogy:
   - **Word 1 (subtract)**: e.g., "man"
   - **Word 2 (add)**: e.g., "woman"
   - **Word 3 (base)**: e.g., "king"
2. Click "🔢 Calculate Analogy"
3. View the result (e.g., "queen") and explore the vector visualization showing:
   - Dotted lines from origin to input words
   - Solid line to the result
   - Vector operation connections between words

### Example Analogies

- `king - man + woman = queen`
- `paris - france + italy = rome`
- `walking - walk + run = running`
- `bigger - big + small = smaller`

## 🧪 Technical Details

### Word Vector Model

- Uses Google's pre-trained Word2Vec model (300 dimensions)
- Vocabulary: ~3 million words and phrases
- Trained on Google News dataset (100 billion words)

### Dimensionality Reduction

- **PCA (Principal Component Analysis)** for reducing 300D vectors to 3D
- **PC1, PC2, PC3** represent the three principal components with highest variance
- Preserves maximum variance while enabling visualization
- Real-time computation for interactive exploration

### Vector Arithmetic

The application performs the mathematical operation: `result = word3 - word1 + word2`

Visualization shows:

1. **Origin vectors** (dotted lines) - from origin to each input word
2. **Operation vectors** (dashed lines) - showing subtraction and addition steps
3. **Result vector** (solid line) - final computed result
4. **Intermediate points** - showing calculation steps

## 🐛 Troubleshooting

• **Model download fails** - Ensure stable internet connection and sufficient disk space (2GB+)
• **Memory errors** - Increase available RAM or use a smaller model variant
• **Import errors** - Verify all dependencies are installed: `pip install -r requirements.txt`
• **Visualization not showing** - Check browser compatibility (Chrome/Firefox recommended)
• **Word not found** - Try different spellings or check if word exists in vocabulary

### Common Issues

- **Threading errors**: Restart the application if model loading hangs
- **Path issues**: Ensure you're running from the project root directory
- **Conda environment**: Activate the correct environment before running

## 💡 Development Notes

### Session State Management

The application uses Streamlit's session state to:

- Cache the loaded Word2Vec model across interactions
- Maintain visualization state between user actions
- Prevent unnecessary model reloading

### Performance Optimizations

- **Lazy loading**: Model loads only when needed
- **Caching**: Gensim handles model caching automatically
- **Error handling**: Graceful degradation for missing words
- **Threading**: Background model loading with progress updates

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional dimensionality reduction methods (t-SNE, UMAP)
- Support for other word embedding models (GloVe, FastText)
- Enhanced visualization features (clustering, word clouds)
- Performance optimizations for larger vocabularies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Original Word2Vec paper
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) - Word2Vec implementation

## 🌟 Acknowledgments

Built with:

- **Streamlit** for the interactive web interface
- **Plotly** for 3D visualizations
- **Gensim** for Word2Vec model handling
- **scikit-learn** for dimensionality reduction
- **NumPy** for numerical computations
