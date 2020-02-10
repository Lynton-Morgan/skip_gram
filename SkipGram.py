import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import os
from datetime import datetime

class SkipGram(object):

    def __init__(self, vocab_length, emb_length):
        self.vocab_length = vocab_length
        self.emb_length = emb_length

    def build_graph(self, vocab_length, emb_length, context_size=None,
            sampling='log-uniform', n_neg_samples=1, unigrams=None, distortion=0.75, tf_seed=None):
        if sampling=='unigram':
            assert unigrams is not None
            assert len(unigrams)==vocab_length

        if context_size is None:
            context_size=1

        g = tf.Graph()
        with g.as_default():
            # word indices
            w = v1.placeholder(tf.int64, shape=(None), name='w')

            # context indices
            c = v1.placeholder(tf.int64, shape=(None, context_size), name='c')

            learning_rate = v1.placeholder_with_default(1e-4, shape=(), name='learning_rate')

            l1_penalty = v1.placeholder_with_default(0.0, shape=(), name='l1_penalty')
            l2_penalty = v1.placeholder_with_default(1.0, shape=(), name='l2_penalty')

            emb_init = v1.initializers.he_normal(seed=tf_seed)
            embeddings = tf.Variable(emb_init(shape=(vocab_length, emb_length)),
                    name='embedding')
            l1_loss = l1_penalty * tf.reduce_mean(tf.abs(embeddings))
            l2_loss = l2_penalty * tf.reduce_mean(tf.square(embeddings))

            w_emb = tf.nn.embedding_lookup(
                    embeddings,
                    w,
                    name='w_emb')
            c_emb = tf.nn.embedding_lookup(
                    embeddings,
                    c,
                    name='c_emb')

            w_emb_reshaped = tf.reshape(w_emb, (-1, 1, emb_length))
            c_logits = tf.reduce_sum(w_emb_reshaped * c_emb, axis=2, name='c_logits')
            loss_normalizer = context_size * tf.reduce_logsumexp(tf.matmul(w_emb, tf.transpose(embeddings)), axis=1)
            loss = tf.reduce_mean(
                    loss_normalizer - tf.reduce_sum(c_logits, axis=1),
                    name='loss')
            regularized_loss = tf.identity(loss + l1_loss + l2_loss, name='regularized_loss')

            if sampling=='uniform':
                sampled_values = tf.random.uniform_candidate_sampler(
                        true_classes=c,
                        num_true=context_size,
                        num_sampled=n_neg_samples,
                        unique=True,
                        range_max=vocab_length)
            elif sampling=='log-uniform':
                sampled_values = tf.random.log_uniform_candidate_sampler(
                        true_classes=c,
                        num_true=context_size,
                        num_sampled=n_neg_samples,
                        unique=True,
                        range_max=vocab_length)
            elif sampling=='unigram':
                sampled_values = tf.random.fixed_unigram_candidate_sampler(
                        true_classes=c,
                        num_true=context_size,
                        num_sampled=n_neg_samples,
                        unique=True,
                        range_max=vocab_length,
                        unigrams=unigrams,
                        distortion=distortion
                        )
            else:
                raise AssertionError('Invalid sampling option')

            sampled_loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                    weights=embeddings,
                    biases=tf.zeros(vocab_length),
                    labels=c,
                    inputs=w_emb,
                    num_sampled=n_neg_samples,
                    num_classes=vocab_length,
                    num_true = context_size,
                    sampled_values=sampled_values,
                    remove_accidental_hits=False),
                    name='sampled_loss')
            training_loss = sampled_loss + l1_loss + l2_loss

            optimizer = v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(training_loss, name='train_op')

        return g


    def show_w_emb(self):
        g = self.build_graph(self.vocab_length, self.emb_length)
        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())

            w = list(range(self.vocab_length))
            return sess.run('w_emb:0', feed_dict={'w:0':w})
       
    def show_c_emb(self):
        n = self.vocab_length
        c = [[(i-1)%n, (i+1)%n] for i in range(self.vocab_length)]
        context_size = len(c[0])

        g = self.build_graph(self.vocab_length, self.emb_length, context_size=context_size)
        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())

            return sess.run('c_emb:0', feed_dict={'c:0':c})

    def return_examples(self):
        n = self.vocab_length

        w = list(range(self.vocab_length))
        c = [[(i-1)%n, (i+1)%n] for i in range(self.vocab_length)]
        context_size = len(c[0])

        g = self.build_graph(self.vocab_length, self.emb_length, context_size=context_size)
        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())

            return sess.run(['w_emb:0', 'c_emb:0', 'c_logits:0'], feed_dict={'w:0':w, 'c:0':c})

    def loss(self, word_indices, context_indices, regularize=False, l1_penalty=0., l2_penalty=1., checkpoint_dir=None, use_batches=True,
            batch_size=1024, n_loss_batches=1000, seed=None):
        g = self.build_graph(self.vocab_length, self.emb_length, tf_seed=seed)
        with g.as_default():
            saver = v1.train.Saver()

        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())

            return self._loss(sess, saver, word_indices, context_indices,
                    regularize=regularize, l1_penalty=l1_penalty, l2_penalty=l2_penalty,
                    checkpoint_dir=checkpoint_dir, use_batches=use_batches, batch_size=batch_size)

    def _loss(self, sess, saver, word_indices, context_indices, regularize=False, l1_penalty=0, l2_penalty=1., checkpoint_dir=None, use_batches=True, batch_size=1024, n_loss_batches=1000):
        assert len(word_indices) == len(context_indices)
        assert type(regularize)==type(True)
        assert type(use_batches)==type(True)

        random = np.random.RandomState()
        
        if regularize:
            loss_name='regularized_loss:0'
        else:
            loss_name='loss:0'

        if checkpoint_dir is not None:
            saver.restore(
                sess, 
                tf.train.latest_checkpoint(checkpoint_dir))

        if use_batches==False:
            feed = {'w:0': word_indices, 'c:0': context_indices,
                    'l1_penalty:0':l1_penalty, 'l2_penalty:0':l2_penalty}
            return sess.run(loss_name, feed_dict=feed)

        elif use_batches==True:
            n_samples = len(word_indices)
            n_batches = len(range(0, n_samples, batch_size))

            n_loss_batches = min(n_loss_batches, n_batches)

            if n_loss_batches == n_batches:
                batches_to_sample = np.arange(n_batches)
            else:
                batches_to_sample = random.choice(n_batches, n_loss_batches, replace=False) 

            losses = np.zeros(n_loss_batches)
            weights = np.zeros(n_loss_batches)

            for i, batch_idx in enumerate(batches_to_sample):
                start = batch_idx * batch_size
                end = start + batch_size

                w, c = word_indices[start:end], context_indices[start:end]
                feed = {'w:0':w, 'c:0':c,
                    'l1_penalty:0':l1_penalty, 'l2_penalty:0':l2_penalty}
                losses[i] = sess.run(loss_name, feed_dict=feed)
                weights[i] = len(w)

            return np.average(losses, None, weights)

    def train(self, word_indices, context_indices, l1_penalty=0., l2_penalty=1., sampling='log-uniform', neg_sample_rate=5,
            unigrams=None, distortion=0.75, learning_rate=1e-4, batch_size=64, n_epochs=5, checkpoint_dir=None,
            load_prev=False, prev_epochs=0, print_reports=False, n_batch_reports=10, n_loss_batches=1000, seed=None):
        assert len(word_indices) == len(context_indices)

        word_indices = np.copy(word_indices)
        context_indices = np.copy(context_indices)

        random = np.random.RandomState(seed)

        n_samples = len(word_indices)
        n_batches = len(range(0, n_samples, batch_size))
        report_at_batches = [int(round(x * n_batches / n_batch_reports)) for x in range(1, n_batch_reports+1)]
        
        context_size = len(context_indices[0])
        n_neg_samples = max(1, int(round(neg_sample_rate * batch_size * context_size)))

        g = self.build_graph(self.vocab_length, self.emb_length, context_size=context_size,
                sampling=sampling, unigrams=unigrams, distortion=distortion, n_neg_samples=n_neg_samples, tf_seed=seed)
        with g.as_default():
            saver = v1.train.Saver()

        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())
            if checkpoint_dir is not None and load_prev:
                saver.restore(
                    sess, 
                    tf.train.latest_checkpoint(checkpoint_dir))

            for epoch in range(1, n_epochs+1):
                for batch_n, j in enumerate(range(0, n_samples, batch_size), 1):
                    w, c = word_indices[j:j+batch_size], context_indices[j:j+batch_size]
                    feed = {'w:0':w, 'c:0':c, 'learning_rate:0':learning_rate,
                            'l1_penalty:0':l1_penalty, 'l2_penalty:0':l2_penalty}

                    _ = sess.run('train_op', feed_dict=feed)

                    if print_reports and batch_n in report_at_batches:
                        loss = sess.run('regularized_loss:0', feed_dict=feed)
                        print(str(datetime.now())+':', 'Epoch %d, batch %d: loss %.4f' % (epoch+prev_epochs, batch_n, loss))

                if print_reports:
                    loss = self._loss(sess, saver, word_indices, context_indices,
                            regularize=True, l1_penalty=l1_penalty, l2_penalty=l2_penalty,
                            use_batches=True, batch_size=batch_size, n_loss_batches=n_loss_batches)
                    print(str(datetime.now())+':', 'Epoch %d: loss %.4f' % (epoch+prev_epochs, loss))

                if checkpoint_dir is not None:
                    saver.save(sess, os.path.join(checkpoint_dir, 'skip_gram_'+str(self.emb_length)), global_step=epoch+prev_epochs)

                if epoch < n_epochs:
                    random.shuffle(word_indices)
                    random.shuffle(context_indices)

    def embed(self, word_indices, checkpoint_dir=None, seed=None):
        g = self.build_graph(self.vocab_length, self.emb_length, tf_seed=seed)
        with g.as_default():
            saver = v1.train.Saver()

        with v1.Session(graph=g) as sess:
            sess.run(v1.global_variables_initializer())

            if checkpoint_dir is None:
                v1.logging.warn('No checkpoint selected. Embedding matrix will be randomly initialized')
            else:
                saver.restore(
                    sess, 
                    tf.train.latest_checkpoint(checkpoint_dir))

            feed={'w:0': word_indices}
            return sess.run('w_emb:0', feed)

