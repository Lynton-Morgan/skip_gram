import numpy as np
import tensorflow as tf

class SkipGram(object):

    def __init__(self, vocab_length, emb_length, tf_seed=0):
        self.vocab_length = vocab_length
        self.emb_length = emb_length
        self.tf_seed = tf_seed

    def build_graph(self, vocab_length, emb_length, context_size=None, n_neg_samples=1):
        if context_size is None:
            context_size=1

        g = tf.Graph()
        with g.as_default():
            # word indices
            w = tf.placeholder(tf.int32, shape=(None), name='w')

            # context indices
            c = tf.placeholder(tf.int32, shape=(None, None), name='c')

            learning_rate = tf.placeholder_with_default(1e-4, shape=(), name='learning_rate')

            emb_init = tf.initializers.he_normal(seed=self.tf_seed)
            embeddings = tf.Variable(emb_init(shape=(vocab_length, emb_length)),
                    name='embedding')

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
            loss_normalizer = tf.reduce_logsumexp(tf.matmul(w_emb, tf.transpose(embeddings)), axis=1)
            loss = tf.reduce_mean(
                    loss_normalizer - tf.reduce_sum(c_logits, axis=1),
                    name='loss')

            sampled_loss = tf.nn.sampled_softmax_loss(
                    weights=embeddings,
                    biases=tf.zeros(vocab_length),
                    labels=c,
                    inputs=w_emb,
                    num_sampled=n_neg_samples,
                    num_classes=vocab_length,
                    num_true = context_size,
                    name='sampled_loss')

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(sampled_loss, name='train_op')

        return g


    def show_w_emb(self):
        g = self.build_graph(self.vocab_length, self.emb_length)
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            w = list(range(self.vocab_length))
            return sess.run('w_emb:0', feed_dict={'w:0':w})
       
    def show_c_emb(self):
        n = self.vocab_length
        c = [[(i-1)%n, (i+1)%n] for i in range(self.vocab_length)]
        context_size = len(c[0])

        g = self.build_graph(self.vocab_length, self.emb_length, context_size=context_size)
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            return sess.run('c_emb:0', feed_dict={'c:0':c})

    def return_examples(self):
        n = self.vocab_length

        w = list(range(self.vocab_length))
        c = [[(i-1)%n, i%n, (i+1)%n] for i in range(self.vocab_length)]
        context_size = len(c[0])

        g = self.build_graph(self.vocab_length, self.emb_length, context_size=context_size)
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            return sess.run(['w_emb:0', 'c_emb:0', 'logits:0'], feed_dict={'w:0':w, 'c:0':c})

    def train(self, word_indices, context_indices, batch_size=64, neg_sample_rate=5, learning_rate=1e-4,
            n_epochs=5, checkpoint_dir=None, load_prev=False, prev_epochs=0, reports_per_epoch=10):
        assert len(word_indices) == len(context_indices)
        n_samples = len(word_indices)
        n_batches = len(range(0, n_samples, batch_size))
        report_at_batches = [int(round(x * n_batches / reports_per_epoch)) for x in range(1, reports_per_epoch+1)]
        
        n_neg_samples = int(round(batch_size * neg_sample_rate))
        context_size = len(context_indices[0])

        g = self.build_graph(self.vocab_length, self.emb_length, n_neg_samples=n_neg_samples, context_size=context_size)
        with g.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint_dir is not None and load_prev==True:
                saver.restore(
                    sess, 
                    tf.train.latest_checkpoint(checkpoint_dir))

            for epoch in range(1, n_epochs+1):
                for batch_n, j in enumerate(range(0, n_samples, batch_size), 1):
                    w, c = word_indices[j:j+batch_size], context_indices[j:j+batch_size]
                    feed = {'w:0':w, 'c:0':c, 'learning_rate:0':learning_rate}
                    _ = sess.run('train_op', feed_dict=feed)

                    if batch_n in report_at_batches:
                        loss = sess.run('loss:0', feed_dict=feed)
                        print('Epoch %d, batch %d: loss %.4f' % (epoch, batch_n, loss))

                if checkpoint_dir is not None:
                    saver.save(sess, checkpoint_dir + "/skip_gram-%d.ckpt" % (epoch+prev_epochs))

    def embed(self, word_indices, checkpoint_dir='./model'):
        g = self.build_graph(self.vocab_length, self.emb_length)
        with g.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            saver.restore(
                sess, 
                tf.train.latest_checkpoint(checkpoint_dir))

            feed={'w:0': word_indices}
            return sess.run('w_emb:0', feed)

