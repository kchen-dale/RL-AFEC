class NetworkConfig(object):
  max_step = 1200

  initial_learning_rate = 0.001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 150
  entropy_weight = 0.1

  save_step = 10
  max_to_keep = 1000

  #Conv
  Conv1D_out = 8
  Dense_out = 256
  batch_norm = False

  optimizer = 'RMSprop'
  #optimizer = 'Adam'

  logit_clipping = 10           #10 or 0, = 0 means logit clipping is disable

class Config(NetworkConfig):
  version = 'RL-RS_1.0'
  # version, model and trace_file name the checkpoint directory: tf_ckpts/<version>-<model>_<trace_file>
  model = 'pure_policy'

  trace_file = 'video_RL_train.txt'
  # test_trace_file = 'video_RL_testing.txt'
  test_trace_file = 'test_t.txt'
  # test_trace_file = 'Emulation_video_test/mother_daughter/mother_daughter_0.15.txt'

  state_length = 90 # the length of loss pattern in one state.
  tm_history = 5

  critical_num = 15
  action_dim = 10
  actions_pool = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

  # Overridden by the train.py flags of the same name
  num_agents = 5
  baseline = 'avg'          #avg, best
  num_iter = 100


def get_config(FLAGS):
  # Use an instance rather than the class itself: instance attributes are pickled
  # by value, so flag overrides survive into worker processes started with 'spawn'
  # (the default on macOS and Windows).
  config = Config()

  for k, v in FLAGS.flag_values_dict().items():
    if hasattr(config, k):
      setattr(config, k, v)

  return config
