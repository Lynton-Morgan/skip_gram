import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from collections import Counter
from string import punctuation
import nltk

from tensorflow.keras.preprocessing.sequence import skipgrams

from SkipGram import SkipGram

def plot_embedding_pca(embeddings, vocab, word_indices, offset=.01, random_state=0):
    fig = plt.figure()

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(embeddings)
    emb_pca = pca.transform(embeddings)

    x = emb_pca[word_indices, 0]
    y = emb_pca[word_indices, 1]

    plt.scatter(x, y, figure=fig)
    for idx in word_indices:
        plt.text(emb_pca[idx, 0] + offset, emb_pca[idx, 1] + offset, vocab[idx],
                figure=fig)

    plt.axis('equal')
    return fig

with open ('shakespeare.txt', 'r') as f:
    text = f.read()
    # Truncate to the main text
    text = text[2772:5552063]
    text = text.lower()

    words = nltk.wordpunct_tokenize(text)

word_counts = Counter()
word_counts.update(words)

vocab_length = len(word_counts)
vocab = [p[0] for p in word_counts.most_common()]

word2int = {p[0]:i for i, p in enumerate(word_counts.most_common())}
int2word = {i:p[0] for i, p in enumerate(word_counts.most_common())}

# skipgrams() assumes 0 is not a word, so some shifting is done
text_indices = np.array([word2int[w] for w in words])
idx_couples = np.array(skipgrams(text_indices+1, vocab_length+1, window_size=4, negative_samples=0.)[0]) - 1
word_indices = idx_couples[:,0]
context_indices = idx_couples[:,1].reshape(-1,1)

sg = SkipGram(vocab_length, emb_length=128)
sg.train(word_indices, context_indices, batch_size=1024, neg_sample_rate=5, n_epochs=3, checkpoint_dir='./model')

emb = sg.embed(list(range(vocab_length)), checkpoint_dir='./model')

test_words = ['man', 'woman', 'men', 'women', 'king', 'queen', 'boy', 'girl']
plot_embedding_pca(emb, vocab, [word2int[w] for w in test_words])
plt.show()
