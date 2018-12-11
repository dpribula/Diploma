from __future__ import print_function, division

from comet_ml import Experiment
from comet_ml import Optimizer

import tensorflow as tf

### Params for nn
state_size = 100
learning_rate = 0.1


class Model():
    def __init__(self, num_classes, steps, session, restore):
        num_steps = steps
        print(num_classes)
        self.sess = session

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.int32, [None, num_steps])
        self.y = tf.placeholder(tf.float32, [None, num_steps])
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.target_x = tf.placeholder(tf.int32, [None, num_steps])
        self.target_y = tf.placeholder(tf.float32, [None, num_steps])

        ##########
        #  MODEL
        ##########

        ###
        ### SHAPING OF DATA
        ###
        inputs_series = tf.split(self.x, num_steps, 1)
        target_label_s = tf.reshape(self.target_y, [-1, num_steps, 1])
        target_label_s = tf.tile(target_label_s, [1, 1, num_classes])
        target_one_hot = tf.one_hot(self.target_x, num_classes)
        rnn_inputs = tf.one_hot(self.x, num_classes)
        y_2 = tf.reshape(self.y, [-1, num_steps, 1])
        rnn_inputs = tf.concat([rnn_inputs, y_2], axis=2)

        # labels_series = tf.unstack(target_x, axis=1)
        ###
        ### FORWARD PASS
        ###
        cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
        # possible to add initial state initial_state=init_state
        output, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, sequence_length=self.seqlen, dtype=tf.float32)
        # TODO embedding one how vector tf.nn.embedding lookup check possible improvements
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

        logits = tf.reshape(tf.matmul(tf.reshape(output, [-1, state_size]), W) + b,
                    [-1, num_steps, num_classes])
        self.predictions_series = tf.sigmoid(logits)

        ###
        ### BACKPROPAGATION
        ###
        target_label_s = target_label_s * target_one_hot
        logits = logits * target_one_hot
        #TODO  check if loss function works properly
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label_s, logits=logits)
        self.total_loss = tf.reduce_sum(losses)
        self.train_step = tf.train.AdagradOptimizer(learning_rate).minimize(self.total_loss)
        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, "/home/dave/projects/savedNetwork/model.ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())

    def run_model_train(self, batch_x, batch_y, batch_target_x, batch_target_y, batch_seq, session):
        return session.run(
            [self.total_loss, self.train_step, self.predictions_series],
            feed_dict={
                self.x: batch_x,
                self.y: batch_y,
                self.target_x: batch_target_x,
                self.target_y: batch_target_y,
                self.seqlen: batch_seq
            })

    def run_model_test(self, batch_x, batch_y, batch_seq):
        # We do not need target as we are not learning on test dataset
        return self.sess.run(self.predictions_series,
                             feed_dict={self.x: batch_x,
                                        self.y: batch_y,
                                        self.seqlen: batch_seq})

    def save_model(self):
        saver_path = self.saver.save(self.sess, "/home/dave/projects/savedNetwork/model.ckpt")
        print("Model was saved into ", saver_path)


