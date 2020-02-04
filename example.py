import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from collections import Counter
from string import punctuation
import nltk

from tensorflow.keras.preprocessing.sequence import skipgrams

from SkipGram import SkipGram
from plot_embedding_pca import plot_embedding_pca

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
unigrams = [p[1] for p in word_counts.most_common()]

word2int = {p[0]:i for i, p in enumerate(word_counts.most_common())}
int2word = {i:p[0] for i, p in enumerate(word_counts.most_common())}
# skipgrams() assumes 0 is not a word, so some shifting is done
text_indices = np.array([word2int[w] for w in words])
idx_couples = np.array(skipgrams(text_indices+1, vocab_length+1, window_size=4, negative_samples=0.)[0]) - 1
word_indices = idx_couples[:,0]
context_indices = idx_couples[:,1].reshape(-1,1)

sg = SkipGram(vocab_length, emb_length=128)
sg.train(word_indices, context_indices, neg_sample_rate=5, sampling='unigram', unigrams=unigrams, learning_rate=1e-3,
        batch_size=512, n_epochs=3, checkpoint_dir='./model', print_reports=True)

emb = sg.embed(list(range(vocab_length)), checkpoint_dir='./model')

n_words=20
n_plots=5
for i in range(n_plots):
    fig = plot_embedding_pca(emb, vocab, [word2int[w] for w in vocab[i*n_words:(i+1)*n_words]], offset=2e-3)
    fig.savefig('pca_example{}.pdf'.format(i+1))

test_words = ['man', 'woman', 'men', 'women', 'king', 'queen', 'he', 'she', 'his', 'her']
fig = plot_embedding_pca(emb, vocab, [word2int[w] for w in test_words])
fig.savefig('pca_example.pdf')
