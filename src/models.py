import tensorflow as tf
import math


class Model:

    def __init__(self, vocabulary_size, num_features):

        self.train_inputs = tf.placeholder(
            tf.int32, shape=[None, num_features])
        self.train_outputs = tf.placeholder(tf.float32, shape=[None, 2])
        embed = self.embed_layer(vocabulary_size, self.train_inputs)
        lstm_out = self.lstm_layer(embed, num_features)
        self.out = self.output_layer(lstm_out)
        self.final_out = self.backtrack(self.out, self.train_outputs)
        

    def embed_layer(self, vocabulary_size, train_inputs):

        embedding_size = 100
        rand = tf.random_uniform([vocabulary_size, embedding_size], -1, 1)
        self.embeddings = tf.Variable(rand)
        embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)
        return embed


    def lstm_layer(self, embed, time_steps):

        self.lstm_size = 200
        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size)
        embed = tf.unstack(embed, time_steps, 1)
        outputs, _ = tf.nn.static_rnn(self.lstm_cell, embed, dtype=tf.float32)
        return outputs[-1]


    def output_layer(self, lstm_out):

        dev = 1.0 / math.sqrt(self.lstm_size)
        te = tf.truncated_normal([self.lstm_size, 2], stddev=dev)
        self.weights1 = tf.Variable(te)
        self.biases1 = tf.Variable(tf.zeros([2]))
        out = tf.matmul(lstm_out, self.weights1) + self.biases1
        return out


    def backtrack(self, out, train_outputs):
        
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                            labels=train_outputs)
        self.loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.01).minimize(self.loss)
        return optimizer
