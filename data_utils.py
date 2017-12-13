"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
All rights reserved.
    
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import sys

TRAIN_DATA_FILE = "dataset/train/Inertial Signals/total_acc_x_train.txt"

def load_training_data():
    """ Returns a matrix of training data.
    shape of result = (n_exp, len)
    """
    data = np.loadtxt(TRAIN_DATA_FILE)
    return data.T


class DataLoader(object):
    def __init__(self, data, batch_size=128, num_steps=1):
        self.batch_size = batch_size
        self.n_data, self.seq_len = data.shape
        #num_batches = (self.n_data // self.batch_size) * self.batch_size
        #TODO(malzantot): needs to be improved to utilize all data
        self._data = data[:self.batch_size , :]
        
        self.num_steps = num_steps
        self._data = self._data.reshape((self.batch_size, self.seq_len, 1))
        self._reset_pointer()

    def _reset_pointer(self):
            self.pointer = 0 

    def reset(self):
        self._reset_pointer()

    def has_next(self):
        return self.pointer + self.num_steps < self.seq_len - 1

    def next_batch(self):
       batch_xs = self._data[:, self.pointer:self.pointer+self.num_steps, :]
       batch_ys = self._data[:, self.pointer+1:self.pointer+self.num_steps+1, :]
       self.pointer = self.pointer + self.num_steps
       return batch_xs, batch_ys
        

