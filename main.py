import streamlit as st
import plotly.graph_objects as go
from word_vectors import WordVectorAnalyzer
from visualization import create_3d_plot, create_analogy_visualization

def main():
    st.title("Word Vector 3D Visualization")

    # Initialize analyzer with loading state
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing Word Vector Analyzer..."):
            st.session_state.analyzer = WordVectorAnalyzer()

    analyzer = st.session_state.analyzer

    # Initialize session state for visualization control
    if 'current_visualization' not in st.session_state:
        st.session_state.current_visualization = None
    if 'current_fig' not in st.session_state:
        st.session_state.current_fig = None

    # Sidebar controls
    st.sidebar.header("📊 Similar Words Visualization")
    word_input = st.sidebar.text_input("Enter a word:", "pet")
    num_similar = st.sidebar.slider("Number of similar words:", 5, 50, 20)

    if st.sidebar.button("🔍 Visualize Similar Words"):
        if word_input:
            with st.spinner("Generating similar words visualization..."):
                similar_words = analyzer.find_similar_words(word_input, num_similar)
                if similar_words:
                    st.session_state.current_fig = create_3d_plot(analyzer, word_input, similar_words)
                    st.session_state.current_visualization = "similar_words"
                else:
                    st.error(f"Word '{word_input}' not found in vocabulary.")
        else:
            st.warning("Please enter a word to visualize.")

    # Vector arithmetic section
    st.sidebar.header("🧮 Vector Arithmetic")
    word1 = st.sidebar.text_input("Word 1 (subtract):", "big")
    word2 = st.sidebar.text_input("Word 2 (add):", "small")
    word3 = st.sidebar.text_input("Word 3 (base):", "biggest")

    if st.sidebar.button("🔢 Calculate Analogy"):
        if word1 and word2 and word3:
            with st.spinner("Calculating vector analogy..."):
                result = analyzer.word_analogy(word1, word2, word3)
                if not result.startswith("Word not found"):
                    st.sidebar.success(f"**Result:** {result}")
                    # Create analogy visualization
                    st.session_state.current_fig = create_analogy_visualization(analyzer, word1, word2, word3, result)
                    st.session_state.current_visualization = "analogy"
                else:
                    st.sidebar.error(result)
        else:
            st.sidebar.warning("Please enter all three words for analogy calculation.")

    # Display current visualization
    if st.session_state.current_fig is not None:
        st.plotly_chart(st.session_state.current_fig, use_container_width=True)
    else:
        st.info("👆 Use the sidebar controls to generate visualizations")
        st.markdown("""
        ### How to use:
        1. **Similar Words**: Enter a word and click 'Visualize Similar Words' to see semantically similar words in 3D space
        2. **Vector Arithmetic**: Enter three words and click 'Calculate Analogy' to see vector arithmetic results (e.g., king - man + woman = queen)
        """)

if __name__ == "__main__":
    main()