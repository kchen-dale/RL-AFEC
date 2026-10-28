from __future__ import print_function

import numpy as np
from absl import app
from absl import flags

from env import Environment
from game import Game
from model_mine import Network
from config import get_config
import time
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')

def sim(config, network, game):
    # evaluate() appends per sample, so start each run with empty result files
    for fn in game.eval_result_files(config):
        open(fn, 'w').close()
    inference_time = 0
    for tm_idx in range(game.tm_cnt - game.tm_history):# not tm_cnt - history + 1, since the last state is only used to calculate reward
        time1 = time.time()
        state = game.get_state(tm_idx)
        policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]

        actions = []
        for i in range(config.critical_num + 1):
            actions.append(policy[i*config.action_dim:(i+1)*config.action_dim].argsort()[-1:])
        time2 = time.time()
        inference_time += time2 - time1
        game.evaluate(config, tm_idx, actions)
    print('total inference time: %.3fs, results written to %s' % (inference_time, ', '.join(game.eval_result_files(config))))

def main(_):
    config = get_config(FLAGS)
    env = Environment(config, is_training=False)
    game = Game(config, env)
    network = Network(config, game.input_state_dims, game.action_dim)

    step = network.restore_ckpt(FLAGS.ckpt)
    if step == 0:
        raise SystemExit('No checkpoint found in %s; train a model first (python3 train.py).' % network.ckpt_dir)
    learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
    
    sim(config, network, game)


if __name__ == '__main__':
    app.run(main)
