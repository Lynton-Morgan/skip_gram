import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embedding_pca(embeddings, vocab, word_indices, random_state=0, offset=.01, width=12, height=8, title='PCA Plot', fit_to_all=True):
    fig = plt.figure()
    fig.set_size_inches(width, height)

    pca = PCA(n_components=2, random_state=random_state)
    if fit_to_all:
        pca.fit(embeddings)
    else:
        pca.fit(embeddings[word_indices])
    emb_pca = pca.transform(embeddings)

    var_ratio = pca.explained_variance_ratio_

    x = emb_pca[word_indices, 0]
    y = emb_pca[word_indices, 1]

    plt.scatter(x, y, figure=fig)
    for idx in word_indices:
        plt.text(emb_pca[idx, 0] + offset, emb_pca[idx, 1] + offset, vocab[idx],
                figure=fig)

    plt.axis('equal')
    plt.xlabel('1st Component ({:.2f}% of variance)'.format(100 * var_ratio[0]))
    plt.ylabel('2nd Component ({:.2f}% of variance)'.format(100 * var_ratio[1]))
    plt.title(title)
    return fig

