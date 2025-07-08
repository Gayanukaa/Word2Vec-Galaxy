import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

def create_3d_plot(analyzer, target_word, similar_words):
    """Create interactive 3D plot of word vectors"""
    if not similar_words:
        st.error(f"No similar words found for '{target_word}'")
        return None

    # Get reduced dimensions
    words, vectors_3d = analyzer.reduce_dimensions(similar_words, method='pca')

    if len(words) == 0:
        st.error(f"No valid words found in vocabulary")
        return None

    # Create colors based on similarity to target word
    colors = []
    for word in words:
        if word == target_word:
            colors.append(1.0)  # Maximum similarity for target word
        else:
            try:
                # Calculate similarity for color intensity
                similarity = analyzer.model.similarity(target_word, word)
                colors.append(similarity)
            except:
                colors.append(0.5)  # Default similarity if calculation fails

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=[12 if word == target_word else 8 for word in words],
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity"),
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=words,
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Similarity: %{marker.color:.3f}<br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f'3D Word Vector Space - Similar to "{target_word}"',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            bgcolor='rgba(0,0,0,0.1)'
        ),
        width=800,
        height=600
    )

    return fig

def create_analogy_visualization(analyzer, word1, word2, word3, result):
    """Visualize word analogy in 3D space"""
    words = [word1, word2, word3, result]

    # Filter out any words not in vocabulary
    valid_words = [word for word in words if word in analyzer.vocab]

    if len(valid_words) < 4:
        # If some words are missing, show an error message
        missing_words = [word for word in words if word not in analyzer.vocab]
        st.error(f"Words not found in vocabulary: {', '.join(missing_words)}")
        return None

    # Get reduced dimensions
    words, vectors_3d = analyzer.reduce_dimensions(valid_words, method='pca')

    # Create colors and labels for different word types
    colors = []
    labels = []
    sizes = []

    word_info = {
        word1: {"color": "red", "label": f"{word1} (subtract)", "size": 12},
        word2: {"color": "blue", "label": f"{word2} (add)", "size": 12},
        word3: {"color": "green", "label": f"{word3} (base)", "size": 12},
        result: {"color": "purple", "label": f"{result} (result)", "size": 15}
    }

    for word in words:
        info = word_info[word]
        colors.append(info["color"])
        labels.append(info["label"])
        sizes.append(info["size"])

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=words,
            textposition="top center",
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Type: %{customdata}<br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<extra></extra>',
            customdata=labels
        )
    ])

    # Add vector arrows to show relationships (simplified)
    # This is a conceptual representation of the vector arithmetic
    if len(vectors_3d) == 4:
        # Find indices
        word1_idx = words.index(word1)
        word2_idx = words.index(word2)
        word3_idx = words.index(word3)
        result_idx = words.index(result)

        # Add lines to show vector relationships
        # Line from word3 to result (should be similar to word2 - word1)
        fig.add_trace(go.Scatter3d(
            x=[vectors_3d[word3_idx, 0], vectors_3d[result_idx, 0]],
            y=[vectors_3d[word3_idx, 1], vectors_3d[result_idx, 1]],
            z=[vectors_3d[word3_idx, 2], vectors_3d[result_idx, 2]],
            mode='lines',
            line=dict(color='orange', width=4, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'Vector Analogy: {word3} - {word1} + {word2} = {result}',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            bgcolor='rgba(0,0,0,0.1)'
        ),
        width=800,
        height=600,
        showlegend=False
    )

    return fig