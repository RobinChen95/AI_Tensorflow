import tensorflow as tf

w_to_id = {'a': 0, 'c': 1, 'h': 2, 'i': 3, 'n': 4}
id_to_w = {0: 'a', 1: 'c', 2: 'h', 3: 'i', 4: 'n'}


class Model():
    def __init__(self, emb_dim, hidden_dim, output_dim, seqlen):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seqlen = seqlen

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self._placeholder()
        self._init_w()
        self._rnn()
        self._loss()
        self._optimize()

    def _placeholder(self):
        self.x_emb = tf.placeholder(dtype=tf.float32, shape=[self.seqlen, self.emb_dim], name="x_emb")

        self.label = tf.placeholder(dtype=tf.int32, shape=[self.seqlen], name="label")

    def _init_w(self):
        self.W_xh = tf.get_variable(shape=[self.hidden_dim, self.emb_dim], name="w_xh", dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_hh = tf.get_variable(shape=[self.hidden_dim, self.hidden_dim], name="w_hh", dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_hy = tf.get_variable(shape=[self.output_dim, self.hidden_dim], name="w_hy", dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

    def _rnn(self):
        self.output = []
        self.h = tf.zeros(shape=[self.hidden_dim, 1], dtype=tf.float32)
        with tf.variable_scope("rnn"):
            for i in range(self.seqlen):
                self.input_ = tf.expand_dims(self.x_emb[i], 1)  # [5, 1]
                self.h = tf.tanh(tf.matmul(self.W_xh, self.input_)
                                 + tf.matmul(self.W_hh, self.h))
                self.o = tf.matmul(self.W_hy, self.h)
                self.output.append(self.o)

    def _loss(self):
        self.logits = tf.squeeze(self.output)
        self.predict = tf.argmax(self.logits, -1)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        self.loss = tf.reduce_mean(self.loss)

    def _optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x_train, y_train, epoch):
        self.sess.run(tf.global_variables_initializer())
        feq = 1
        for i in range(epoch):
            feed_dict = {self.x_emb: x_train, self.label: y_train}

            _, loss, predict_id = self.sess.run([self.train_op, self.loss, self.predict], feed_dict)

            predict = [id_to_w[predict_id[j]] for j in range(len(predict_id))]
            if i % feq == 0:
                print('epoch num={} loss={} predict={}'.format(i, loss, predict))

    def test(self, input):
        feed_dict = {self.x_emb: input, self.label: [0] * self.seqlen}

        predict = self.sess.run(self.predict, feed_dict)
        return predict


input_word = 'china'

rnn = Model(5, 3, 5, seqlen=len(input_word) - 1)

x_train = []
y_train = []
for i in range(len(input_word)):
    emb = [0.0] * len(w_to_id)
    emb[w_to_id[input_word[i]]] = 1.0
    if i < len(input_word) - 1:
        x_train.append(emb)
    if i > 0:
        y_train.append(w_to_id[input_word[i]])

rnn.train(x_train=x_train, y_train=y_train, epoch=100)
predict_id = rnn.test(input=x_train)
predict = [id_to_w[predict_id[i]] for i in range(len(predict_id))]
print(predict)
