import streamlit as st

from visualization import create_3d_plot, create_analogy_visualization
from word_vectors import WordVectorAnalyzer


def main():
    """Main Streamlit application"""
    st.title("Word Vector 3D Visualization")

    # Load model once and store in session state
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = WordVectorAnalyzer()

    analyzer = st.session_state.analyzer

    st.info("👆 Use the sidebar controls to generate visualizations")
    st.markdown(
        """
    ### How to use:
    1. **Similar Words**: Enter a word and click 'Visualize Similar Words' to see semantically similar words in 3D space
    2. **Vector Arithmetic**: Enter three words and click 'Calculate Analogy' to see vector arithmetic results

    ### Example analogies:
    - king - man + woman = queen
    - walking - walk + run = running
    """
    )

    # Sidebar controls for similar words
    st.sidebar.header("📊 Similar Words")
    word_input = st.sidebar.text_input("Enter a word:", "pet")
    num_similar = st.sidebar.slider("Number of similar words:", 5, 50, 20)

    if st.sidebar.button("🔍 Visualize Similar Words"):
        if word_input:
            similar_words = analyzer.find_similar_words(word_input, num_similar)
            if similar_words:
                fig = create_3d_plot(analyzer, word_input, similar_words)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # Vector arithmetic section
    st.sidebar.header("🧮 Vector Arithmetic")
    word1 = st.sidebar.text_input("Word 1 (subtract):", "man")
    word2 = st.sidebar.text_input("Word 2 (add):", "woman")
    word3 = st.sidebar.text_input("Word 3 (base):", "king")

    if st.sidebar.button("🔢 Calculate Analogy"):
        if word1 and word2 and word3:
            result = analyzer.word_analogy(word1, word2, word3)
            st.sidebar.success(f"Result: {result}")

            fig = create_analogy_visualization(analyzer, word1, word2, word3, result)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
