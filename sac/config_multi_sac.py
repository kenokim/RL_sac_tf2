import tensorflow as tf
class config_multi_sac:
    def __init__(self, use_baseline):
        self.env_name = "Extra"
        self.record = True 
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1
        self.discrete = False
        self.out_activation = None
        # model and training config
        self.batchnumber            = 256 # number of batches trained on
        self.batch_size             = 100000 # number of steps used to compute each policy update
        self.max_ep_len             = 100000 # maximum episode length
        self.learning_rate          = 0.0003
        self.gamma                  = 0.99 # the discount factor
        self.normalize_advantage    = True 
        self.lambdas                = 0.95
        self.usebatch               = True
        self.value_batch            = False
        self.num_val_epochs         = 1
        self.action_space_dims      = 3
        self.action_constant = 1
        self.action_bias = 0
        self.update_rate = 0.005                                                                                                                                                                     
        self.alphas = 1
        self.clip_max = 2
        self.clip_min = -20
        self.initial_batch = 10**4
        self.use_checkpoint = True
        ### New
        self.use_replay_buffer = True
        self.continuity = True
        self.on_policy = False

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu 
        self.replay_buffer_size = 10**6
        self.batch_total = 30

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size