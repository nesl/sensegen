"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
All rights reserved Networked and Embedded Systems Lab (NESL), UCLA.
    
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

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
        
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.rnn_size) for _ in range(self.num_layers)], state_is_tuple=True)
        self.init_state = multi_rnn_cell.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
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
        # self.final_c_state = final_state.c
        # self.final_h_state = final_state.h
        if self.is_training:
            self.optimizer = tf.train.AdamOptimizer()
            self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.y_holder))
            self.train_op = self.optimizer.minimize(self.loss)
            
     
    def train_for_epoch(self, sess, data_loader):
        assert self.is_training, "Must be training model"
        cur_state = sess.run(self.init_state)
        data_loader.reset()
        epoch_loss = []
        while data_loader.has_next():
            batch_xs, batch_ys = data_loader.next_batch()
            batch_xs = batch_xs.reshape((self.batch_size, self.num_steps, 1))
            batch_ys = batch_ys.reshape((self.batch_size, self.num_steps, 1))
            _, batch_loss_, new_state_ = sess.run(
                [self.train_op, self.loss, self.final_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.y_holder: batch_ys,
                    self.init_state: cur_state,
                })
            cur_state = new_state_
            epoch_loss.append(batch_loss_)
         
            
        return np.mean(epoch_loss)
    
    
    def predict(self,sess, seq_len=1000):
        assert not self.is_training, "Must be testing model"
        cur_state = sess.run(self.init_state)
        preds = []
        preds.append(np.random.uniform())
        for step in range(seq_len):
            batch_xs = np.array(preds[-1]).reshape((self.batch_size, self.num_steps, 1))
            new_pred_, new_state_ = sess.run(
                [self.preds, self.final_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.init_state: cur_state
                }
            )
            preds.append(new_pred_[0][0])
            cur_state = new_state_
        return preds[1:]
    
class MDNModel(object):
    def __init__(self, config, is_training=True):
        self.batch_size = config.batch_size
        self._config = config
        self.rnn_size = config.rnn_size
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.is_training = is_training
        self.num_steps = config.num_steps
        self.num_mixtures = config.num_mixtures
        self.n_gmm_params = self.num_mixtures * 3
        self.learning_rate = config.learning_rate
        with tf.variable_scope('mdn_model', reuse=(not self.is_training)):
            self._build_model()
    

    def _build_model(self):
        """ Build the MDN Model"""
        self.x_holder = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, 1 ], name="x")
        self.y_holder = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, 1], name="y")
        
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.rnn_size) for _ in range(self.num_layers)], state_is_tuple=True)
        self.init_state = multi_rnn_cell.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                     inputs=self.x_holder,
                                                     initial_state=self.init_state)
        
        w1 = tf.get_variable('w1', shape=[self.rnn_size, self.hidden_size], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.2))
        b1 = tf.get_variable('b1', shape=[self.hidden_size], dtype=tf.float32, 
                            initializer=tf.constant_initializer())
        h1 = tf.nn.sigmoid(tf.matmul(tf.reshape(rnn_outputs, [-1, self.rnn_size]), w1) + b1)
        w2 = tf.get_variable('w2', shape=[self.hidden_size, self.n_gmm_params], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.2))
        b2 = tf.get_variable('b2', shape=[self.n_gmm_params], dtype=tf.float32,
                             initializer=tf.constant_initializer())
        gmm_params = tf.matmul(h1, w2) + b2
        print(gmm_params)
        mu_ = gmm_params[:, : self.num_mixtures]
        sigma_= gmm_params[:,  self.num_mixtures: 2* self.num_mixtures]
        pi_ = gmm_params[:,  2*self.num_mixtures:]
        print(pi_)
        self.mu = mu_
        self.sigma = tf.exp(sigma_ / 2.0)
        self.pi = tf.nn.softmax(pi_)
        print(self.mu)
        print(self.sigma)
        # self.final_c_state = final_state.c
        # self.final_h_state = final_state.h
        if self.is_training:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.y_holder))
            print(self.y_holder)
            
            mixture_p = tf.compat.v1.distributions.Normal(self.mu, self.sigma).prob(tf.reshape(self.y_holder,(-1,1)))
            mixture_p = tf.multiply(self.pi, mixture_p)
            output_p = tf.reduce_sum(mixture_p, reduction_indices=1, keep_dims=True)
            log_output_p = tf.log(output_p)
            mean_log_output_p = tf.reduce_mean(log_output_p)
            self.loss = -mean_log_output_p   
            self.train_op = self.optimizer.minimize(self.loss)
            
     
    def train_for_epoch(self, sess, data_loader):
        assert self.is_training, "Must be training model"
        cur_state = sess.run(self.init_state)
        data_loader.reset()
        epoch_loss = []
        while data_loader.has_next():
            batch_xs, batch_ys = data_loader.next_batch()
            batch_xs = batch_xs.reshape((self.batch_size, self.num_steps, 1))
            batch_ys = batch_ys.reshape((self.batch_size, self.num_steps, 1))
            _, batch_loss_, new_state_ = sess.run(
                [self.train_op, self.loss, self.final_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.y_holder: batch_ys,
                    self.init_state: cur_state,
                })
            cur_state = new_state_
            epoch_loss.append(batch_loss_)
         
            
        return np.mean(epoch_loss)
    
    
    def predict(self,sess, seq_len=1000):
        assert not self.is_training, "Must be testing model"
        cur_state = sess.run(self.init_state)
        preds = []
        preds.append(np.random.uniform())
        for step in range(seq_len):
            batch_xs = np.array(preds[-1]).reshape((self.batch_size, self.num_steps, 1))
            mu_, sigma_, pi_, new_state_ = sess.run(
                [self.mu, self.sigma, self.pi, self.final_state],
                feed_dict = {
                    self.x_holder: batch_xs,
                    self.init_state: cur_state
                }
            )

            # chose one
            select_mixture = np.random.choice(self.num_mixtures, p=pi_[0])
            #new_pred_  = np.random.normal(loc=mu_[0
            new_pred_ = np.random.normal(loc=mu_[0][select_mixture], scale=sigma_[0][select_mixture])
            preds.append(new_pred_)
            cur_state = new_state_
        return preds[1:]
            
            

      



