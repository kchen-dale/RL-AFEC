from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import inspect
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class Network():
    def __init__(self, config, input_dims, action_dim, master=True):
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.critical_num = config.critical_num
        # The checkpoint directory is named after this, so keep it stable for existing checkpoints.
        self.model_name = config.version+'-'\
                            +config.model+'_'\
                            +config.trace_file

        self.create_policy_model(config)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                config.initial_learning_rate,
                config.learning_rate_decay_step,
                config.learning_rate_decay_rate,
                staircase=True)

        if config.optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        elif config.optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        if master:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.policy_model)
            self.ckpt_dir = './tf_ckpts/'+self.model_name
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=config.max_to_keep)
            self.writer = tf.compat.v2.summary.create_file_writer('./logs/%s' % self.model_name)

    def create_policy_model(self, config):
        tf.keras.backend.set_floatx('float32')
        inputs = tf.keras.Input(shape=(self.input_dims[0], self.input_dims[1]))# not including the batch size

        Conv1D_1_loss_pattern = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        x_1 = Conv1D_1_loss_pattern(inputs)

        if config.batch_norm:
            x_1 = tf.keras.layers.BatchNormalization()(x_1)

        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        x_1 = tf.keras.layers.Flatten()(x_1)

        Dense1_1 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_1(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        Dense1_2 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_2(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        Dense2_1 = tf.keras.layers.Dense(self.action_dim * (self.critical_num+1)) #output layer: 10*(K+1)
        logits = Dense2_1(x_1)
        if config.logit_clipping > 0:
            logits = config.logit_clipping*tf.keras.activations.tanh(logits)

        self.policy_model = tf.keras.models.Model(inputs, logits)

    def policy_loss_fn_product_multiactions(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-45):
        # actions dimension: ( Batchsize, (K+1)*10 )
        #calculate policy
        policy = [] #shape ( 1, 10*(critical_num+1) )
        for i in range(self.critical_num + 1):
            policy.append(tf.nn.softmax(tf.multiply(logits[:, self.action_dim*i:self.action_dim*(i+1)], 1.0/math.sqrt(float(self.action_dim)))))
        policy = tf.stack(policy, axis=1)# [Batchsize, (critical_num+1), 10]
        policy = tf.reshape(policy, logits.shape)

        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]                  #policy shape = [batch_size, action_dim]

        entropy = -tf.reduce_sum(tf.multiply(policy, tf.math.log(tf.maximum(policy, log_epsilon))), 1, keepdims=True)       #shape=[batch_size, 1]
        policy = tf.expand_dims(policy, -1)     #policy:[Batch_size, 160, 1], actions: [Batch_size, 1, 160]
        actions = tf.expand_dims(actions, 1)

        policy_loss = tf.squeeze(tf.matmul(actions, tf.math.log(tf.maximum(policy, log_epsilon)))) #[Batch_size,]
        """ '-'advantages means gradient asdcend. same for entropy """
        policy_loss = tf.expand_dims(policy_loss, -1)
        assert advantages.shape == policy_loss.shape
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                                   #[batch_size, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)

        return policy_loss, entropy

    @tf.function
    def policy_train(self, inputs, actions, advantages, entropy_weight=0.01):
        # Tracks the variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            logits = self.policy_model(inputs, training=True)
            policy_loss, entropy = self.policy_loss_fn_product_multiactions(logits, actions, advantages, entropy_weight)

        gradients = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))

        return entropy, gradients, policy_loss

    @tf.function
    def policy_predict(self, inputs):
        logits = self.policy_model(inputs, training=False)
        policy = [] #shape ( 1, 10*(critical_num+1) )
        for i in range(self.critical_num + 1):
            policy.append(tf.nn.softmax(logits[:, self.action_dim*i:self.action_dim*(i+1)]))
        policy = tf.stack(policy, axis=1)
        policy = tf.reshape(policy, logits.shape)
        return policy

    def restore_ckpt(self, checkpoint=''):
        if checkpoint == '':
            checkpoint = self.manager.latest_checkpoint
        else:
            checkpoint = self.ckpt_dir+'/'+checkpoint

        self.ckpt.restore(checkpoint).expect_partial()
        if checkpoint:
            step = int(self.ckpt.step)
            print("Restored from {}".format(checkpoint), step)
        else:
            step = 0
            print("Initializing from scratch.")

        return step

    def save_ckpt(self, _print=False):
        save_path = self.manager.save()
        if _print:
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def inject_summaries(self, summary_dict, step):
        with self.writer.as_default():
            for summary in summary_dict:
                tf.summary.scalar(summary, summary_dict[summary], step=step)
            self.writer.flush()

    def save_hyperparams(self, config):
        fp = self.ckpt_dir + '/hyper_parameters'

        hparams = {k:v for k, v in inspect.getmembers(config)
            if not k.startswith('__') and not callable(v)}

        if os.path.exists(fp):
            f = open(fp, 'r')
            match = True
            for line in f:
                idx = line.find('=')
                if idx == -1:
                    continue
                k = line[:idx-1]
                v = line[idx+2:-1]
                # a key missing from the current config (e.g. a removed option) also counts as a change
                if k not in hparams or v != str(hparams[k]):
                    match = False
                    print(k, v, hparams.get(k))
                    break
            f.close()
            if match:
                return

            f = open(fp, 'a')
        else:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            f = open(fp, 'w+')

        for k, v in hparams.items():
            f.writelines(k + ' = ' + str(v) + '\n')
        f.writelines('\n')
        print("Save hyper parameters: %s" % fp)
        f.close()
