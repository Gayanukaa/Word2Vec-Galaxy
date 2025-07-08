import streamlit as st
import plotly.graph_objects as go
from word_vectors import WordVectorAnalyzer
from visualization import create_3d_plot

def main():
    st.title("Word Vector 3D Visualization")

    # Initialize analyzer with loading state

    # Sidebar controls
    st.sidebar.header("Controls")
    word_input = st.sidebar.text_input("Enter a word:", "pet")
    num_similar = st.sidebar.slider("Number of similar words:", 5, 50, 20)

    # Vector arithmetic section
    st.sidebar.header("Vector Arithmetic")
    word1 = st.sidebar.text_input("Word 1 (subtract):", "big")
    word2 = st.sidebar.text_input("Word 2 (add):", "small")
    word3 = st.sidebar.text_input("Word 3 (base):", "biggest")

    if st.sidebar.button("Calculate Analogy"):
        result = analyzer.word_analogy(word1, word2, word3)
        st.sidebar.write(f"Result: {result}")

    # Main visualization
    if word_input:
        similar_words = analyzer.find_similar_words(word_input, num_similar)
        fig = create_3d_plot(analyzer, word_input, similar_words)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()