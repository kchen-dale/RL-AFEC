from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from RS_decode_video import decode
from get_vq import get_video_quality

class Game(object):
    def __init__(self, config, env):
        self.tm_history = config.tm_history
        self.critical_num = config.critical_num

        self.action_dim = env.action_dim
        self.actions_pool = config.actions_pool

        self.loss_pattern = env.loss_pattern
        
        self.input_state_dims = env.input_state_dims # [state_length, tm_history]
        self.tm_cnt = env.tm_cnt

        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.train_tm_cnt = len(self.tm_indexes)
        
        self.generate_loss_delay()
        self.baseline = {}

    def get_state(self, tm_idx):

        return self.state_data[tm_idx] # [state_length, tm_history]

    def generate_loss_delay(self):
        self.state_loss = np.zeros(([self.train_tm_cnt] + self.input_state_dims), dtype=np.float32) #[N, state_length, tm_history]
        for i in range(self.train_tm_cnt):
            tm_idx = i+self.tm_history-1
            for h in range(self.tm_history):
                self.state_loss[i,:,h] = self.loss_pattern[tm_idx-h]
        self.state_data = self.state_loss


    def penalty(self, vmaf):
        if vmaf < 40:
            return vmaf/(1 + 0.1*(80 - vmaf))
        elif vmaf < 60:
            return 0.6*vmaf - 16
        else:
            return 3*vmaf - 160
    
    def simulate(self, tm_idx, actions):
        """Apply FEC actions to the next loss pattern; return (vmaf, bw_waste, loss_pattern_after)."""
        loss_pattern_current = self.state_loss[tm_idx+1,:,0]# the current action is for the next state
        critical_frames_num = self.critical_num
        percentage = []
        for i in range(self.critical_num + 1):
            percentage.append(self.actions_pool[actions[i][0]]) # actions: [array([0]), array([3]), ...]
        loss_pattern_after, bw_waste = decode(loss_pattern_current.squeeze(), percentage, critical_frames_num)
        loss_pattern_after = loss_pattern_after.reshape((1, self.input_state_dims[0]))
        vmaf = get_video_quality(loss_pattern_after)

        return vmaf, bw_waste, loss_pattern_after

    def reward(self, tm_idx, actions):
        alpha = 0.5
        vmaf_target = np.array([[80.0]])
        bw_lowerbound = np.array([[30.0]]) #
        vmaf, bw_waste, _ = self.simulate(tm_idx, actions)
        if vmaf > vmaf_target:
            r = 5*(vmaf_target/80) - alpha * (bw_waste / 1500) / bw_lowerbound
        else:
            # r = (self.penalty(vmaf)/vmaf_target)*5 - alpha * (bw_waste / 1500) / bw_lowerbound
            r = - vmaf_target / (self.penalty(vmaf) + 1)

        return r

    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    def eval_result_files(self, config):
        file_name = config.test_trace_file.split('/')[-1]
        return "eval_results_"+file_name, "loss_pattern_after_"+file_name

    def evaluate(self, config, tm_idx, actions):
        eval_file, loss_file = self.eval_result_files(config)
        f = open(eval_file, "a+")
        f1 = open(loss_file, "a+")

        vmaf, bw_waste, loss_pattern_after = self.simulate(tm_idx, actions)
        vmaf = vmaf.item() # np.asscalar was removed in NumPy 1.23
        for i in range(self.critical_num + 1):
            actions[i] = actions[i].item()
        f.write(str(tm_idx+1)+" "+str(actions)+" "+str(vmaf)+" "+str(bw_waste)+ "\n")
        # one space-separated line per sample (str() of a numpy array wraps across lines)
        f1.write(" ".join("%f" % v for v in loss_pattern_after.squeeze()[:30]) + "\n")
        f.close()
        f1.close()
