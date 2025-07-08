import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def create_3d_plot(analyzer, target_word, similar_words):
    """Create interactive 3D plot of word vectors"""

    # Get reduced dimensions
    words, vectors_3d = analyzer.reduce_dimensions(similar_words, method='pca')

    # Create colors based on similarity to target word
    colors = []
    for word in words:
        if word == target_word:
            colors.append('red')
        else:
            # Calculate similarity for color intensity
            similarity = analyzer.model.similarity(target_word, word)
            colors.append(similarity)

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity")
            ),
            text=words,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
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
            zaxis_title='Dimension 3'
        ),
        width=800,
        height=600
    )

    return fig

def create_analogy_visualization(analyzer, word1, word2, word3, result):
    """Visualize word analogy in 3D space"""
    words = [word1, word2, word3, result]
    words, vectors_3d = analyzer.reduce_dimensions(words, method='pca')

    # Create arrows showing the vector arithmetic
    fig = go.Figure()

    # Add word points
    fig.add_trace(go.Scatter3d(
        x=vectors_3d[:, 0],
        y=vectors_3d[:, 1],
        z=vectors_3d[:, 2],
        mode='markers+text',
        marker=dict(size=10, color=['blue', 'green', 'red', 'purple']),
        text=words,
        textposition="top center"
    ))

    # Add vector arrows (simplified representation)
    # This would require more complex math for actual vector visualization

    fig.update_layout(
        title=f'Vector Analogy: {word3} - {word1} + {word2} = {result}',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        )
    )

    return fig