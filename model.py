"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import numpy as np
import tensorflow as tf

import model_utils

class ModelConfig(object):
    def __init__(self):
        self.num_layers = 3  # Number of LSTM layers
        self.rnn_size = 128  # Number of LSTM units
        self.hidden_size = 32  # Number of hidden layer units
        self.num_mixtures = 24
        self.batch_size = 4
        self.num_steps = 10
        self.dropout_rate = 0.5  # Dropout rate
        self.learning_rate = 0.001  # Learning rate

class RNNModel(object):
    def __init__(self, config, is_training=True):
        self.batch_size = config.batch_size
        self._config = config
        self.rnn_size = config.rnn_size
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.is_training = is_training
        self.num_steps = config.num_steps
        self.n_gmm_params = self._config.num_mixtures * 3
        with tf.variable_scope('rnn_model', reuse=(not self.is_training)):
            self._build_model()
    

    def _build_model(self):
        """ Build the MDN Model"""
        self.x_holder = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, 1 ], name="x")
        self.y_holder = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, 1], name="y")
        
        rnn_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.rnn_size, forget_bias=1.0, state_is_tuple=True)
        #TODO(malzantot): fix
        #rnn_cell = tf.contrib.rnn.MultiRNNCell(
        #    [rnn_cell for _ in range(self.num_layers)], state_is_tuple=True)
        self.zero_state = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.state_c_holder = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size])
        self.state_h_holder = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size])
        self.init_state = tf.nn.rnn_cell.LSTMStateTuple(c=self.state_c_holder,
                                                        h=self.state_h_holder)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                     inputs=self.x_holder,
                                                     initial_state=self.init_state)
        
        w1 = tf.get_variable('w1', shape=[self.rnn_size, self.hidden_size], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.2))
        b1 = tf.get_variable('b1', shape=[self.hidden_size], dtype=tf.float32, 
                            initializer=tf.constant_initializer())
        h1 = tf.nn.sigmoid(tf.matmul(tf.reshape(rnn_outputs, [-1, self.rnn_size]), w1) + b1)
        w2 = tf.get_variable('w2', shape=[self.hidden_size, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.2))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer())
        output_fc = tf.matmul(h1, w2) + b2
        self.preds = tf.reshape(output_fc, [self.batch_size, self.num_steps, 1])
        self.final_c_state = final_state.c
        self.final_h_state = final_state.h
        if self.is_training:
            self.optimizer = tf.train.AdamOptimizer()
            self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.y_holder))
            self.train_op = self.optimizer.minimize(self.loss)
            
     
    def train_for_epoch(self, sess, data_loader):
        assert self.is_training, "Must be training model"
        cur_c, cur_h = sess.run(self.zero_state)
        data_loader.reset()
        epoch_loss = []
        while data_loader.has_next():
            batch_xs, batch_ys = data_loader.next_batch()
            batch_xs = batch_xs.reshape((self.batch_size, self.num_steps, 1))
            batch_ys = batch_ys.reshape((self.batch_size, self.num_steps, 1))
            _, batch_loss_, c_state_, h_state_ = sess.run(
                [self.train_op, self.loss, self.final_c_state, self.final_h_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.y_holder: batch_ys,
                    self.state_c_holder: cur_c,
                    self.state_h_holder: cur_h,
                })
            cur_c = c_state_
            cur_h = h_state_
            epoch_loss.append(batch_loss_)
         
            
        return np.mean(epoch_loss)
    
    
    def predict(self,sess, seq_len=1000):
        assert not self.is_training, "Must be testing model"
        cur_c, cur_h = sess.run(self.zero_state)
        preds = []
        preds.append(np.random.uniform())
        for step in range(seq_len):
            batch_xs = np.array(preds[-1]).reshape((self.batch_size, self.num_steps, 1))
            new_pred_, c_state_, h_state_ = sess.run(
                [self.preds, self.final_c_state, self.final_h_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.state_c_holder: cur_c,
                    self.state_h_holder: cur_h
                }
            )
            preds.append(new_pred_[0][0])
            cur_c = c_state_
            cur_h = h_state_
        return preds[1:]
            
            

      



