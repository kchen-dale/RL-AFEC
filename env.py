from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Traffic(object):
    def __init__(self, config, data_dir='./data/', is_training=False):
        if is_training:
            self.trace_file = data_dir + config.trace_file

        else:
            self.trace_file = data_dir + config.test_trace_file

        self.tm_history = config.tm_history
        self.state_length = config.state_length
        self.load_traffic()

    def load_traffic(self):
        print('[*]Load loss pattern', self.trace_file)
        f_trace = open(self.trace_file, 'r')
        loss_pattern = []
        for line in f_trace:
            pattern = [float(v) for v in line.split()]
            line_length = len(pattern)
            assert line_length == self.state_length, (line_length, self.state_length)
            loss_pattern.append(pattern)
        f_trace.close()
        self.loss_pattern = np.array(loss_pattern)
        pattern_shape = self.loss_pattern.shape #[pattern_cnt, config.state_length]
        self.pattern_cnt = pattern_shape[0]

        self.input_state_dims = [pattern_shape[1], self.tm_history]


class Environment(object):
    def __init__(self, config, is_training=False):
        self.data_dir = './data/'
        self.traffic = Traffic(config, self.data_dir, is_training=is_training)
        self.tm_history = config.tm_history
        self.action_dim = config.action_dim
        self.loss_pattern = self.traffic.loss_pattern
        self.input_state_dims = self.traffic.input_state_dims
        self.tm_cnt = self.traffic.pattern_cnt
