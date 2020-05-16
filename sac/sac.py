import os
import sys
import time
import numpy as np
import tensorflow as tf
#from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import matplotlib.pyplot as plt
import datetime
import tensorflow_probability as tfp
tfd = tfp.distributions  
import gym
import config_multi_sac
from model import *
from utils import *
from sac_train import *
from actor import *




class sac(object):
  def __init__(self, env, config):
    self.env = env
    self.config = config
    if self.config.use_replay_buffer:
      self.replay_buffer = []
    self.obs_len = len(self.env.reset())
    self.loss_ = []
    self.build()



  def build(self):
    ep = self.config.num_val_epochs
    ol = self.obs_len
    self.model = polinet(self.env, ol, len(self.env.action_space.sample()), continuity = self.config.continuity).model
    if self.config.on_policy:
      self.valuenet = valuenet(self.env, ep)
    else:
      self.qnet, self.qnet2, self.tar_qnet, self.tar_qnet2 = valuenet(self.env, ep),valuenet(self.env, ep), valuenet(self.env, ep), valuenet(self.env, ep)
    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.learning_rate, decay_steps = 100000, decay_rate = 0.96)
    self.opts = tf.keras.optimizers.Adam(self.lr_schedule, clipvalue = 5)
    #self.ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizers = self.opts, net = )
    checkpoint_path = "./checkpoints"
    self.ckpt = tf.train.Checkpoint(step = tf.Variable(1),
                                    model = self.model,
                                    qnet_model = self.qnet.model,
                                    qnet2_model = self.qnet2.model,
                                    tar_qnet_model = self.tar_qnet.model,
                                    tar_qnet2_model = self.tar_qnet2.model,
                                    opts = self.opts,
                                    qnet_opt = self.qnet.opt,
                                    qnet2_opt = self.qnet2.opt,
                                    tar_qnet_opt = self.tar_qnet.opt,
                                    tar_qnet2_opt = self.tar_qnet2.opt)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=10)
    self.sampling(100)



  def sampling(self, num_episode = None):
    paths = sample_path(self.env, self.model, self.config.batch_size, self.config.max_ep_len, self.config.continuity, num_episodes = num_episode)
    if self.config.use_replay_buffer:
      self.replay_buffer.extend(paths)
      crit = len(self.replay_buffer) - self.config.replay_buffer_size
      if crit > 0:
        self.replay_buffer = self.replay_buffer[crit:]
    return paths

  def sample_action(self, obs):
    try:
        mu, log_sig = self.model(tf.expand_dims(obs, axis = 0))
    except:
        mu, log_sig = self.model(obs)
    action, _ = action_log_pi(mu, tf.math.exp(log_sig))
    return action

  
  def train_step2(self):
    self.paths = self.sampling(self.config.batch_total)
    mask = np.random.choice(len(self.replay_buffer), self.config.batchnumber)
    batch = np.take(self.replay_buffer, mask, axis = 0)
    _ = train_critic(batch, self.model, self.qnet, self.qnet2, self.tar_qnet, self.tar_qnet2)
    obs, _, _, _, _ = path_reshape(batch)

    shape = [self.config.batchnumber, self.config.action_space_dims]

    dist = tfd.MultivariateNormalDiag(loc = np.zeros(shape), scale_diag = np.ones(shape))
    noise = dist.sample()
    prob = dist.log_prob(noise)
    log_prob = tf.cast(prob, tf.float32)
    noise = tf.cast(noise, tf.float32)

    pt = policy_train()
    shape = [self.config.batchnumber, self.config.action_space_dims]
    loss_, mean_q, mean_entropy = pt(tf.cast(obs, tf.float32), self.model, self.qnet, self.opts, noise, log_prob)
    update_network_weights(self.config.update_rate, self.qnet.model, self.tar_qnet.model)
    update_network_weights(self.config.update_rate, self.qnet2.model, self.tar_qnet2.model)
    return loss_, mean_q, mean_entropy

  def train_step(self):
    self.paths = self.sampling(self.config.batch_total)
    mask = np.random.choice(len(self.replay_buffer), self.config.batchnumber)
    batch = np.take(self.replay_buffer, mask, axis = 0)
    _ = train_value(batch, self.model, self.qnet, self.qnet2, self.tar_qnet, self.tar_qnet2)
    mask = np.random.choice(len(self.replay_buffer), self.config.batchnumber)
    batch = np.take(self.replay_buffer, mask, axis = 0)
    obs, _, _, _, _ = path_reshape(batch)

    #dist = tfd.MultivariateNormalDiag(loc = np.zeros(shape), scale_diag = np.ones(shape))
    #noise = dist.sample()
    #prob = dist.log_prob(noise)
    #log_prob = tf.cast(prob, tf.float32)
    #noise = tf.cast(noise, tf.float32)

    noise = sampler(self.config.batchnumber, self.config.action_space_dims)
    noise = tf.cast(noise, tf.float32)

    pt = policy_training()
    shape = [self.config.batchnumber, self.config.action_space_dims]
    loss_, mean_q, mean_entropy = pt(tf.cast(obs, tf.float32), self.model, self.qnet, self.opts, noise)#, log_prob)
    update_network_weights(self.config.update_rate, self.qnet.model, self.tar_qnet.model)
    update_network_weights(self.config.update_rate, self.qnet2.model, self.tar_qnet2.model)
    return loss_, mean_q, mean_entropy



  def train(self, n):
    if self.config.use_checkpoint:
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print('\nRestored checkpoint of step : {}'.format(int(self.ckpt.step)))

    for i in range(n):
      print('\n')
      print(i,'th iteration ...')
      loss_, mean_q, mean_entropy = self.train_step()
      print('Loss is :', loss_)
      print('Mean q is :', mean_q)
      print('Mean entropy is :', mean_entropy)
      print(len(self.replay_buffer))
      returns = self.estimator()
      if i/10 == i//10 :
        pass
      if self.config.use_checkpoint:
          
          if i % 1 == 0:
              self.ckpt_manager.save()
          self.ckpt.step.assign_add(1)
          print('Checkpoint step is :', int(self.ckpt.step))
          self.loss_.append([int(self.ckpt.step), returns])
      else:
          self.loss_append([i, returns])
    print('done!')


  def estimator(self):
    rwd = 0
    for i in range(len(self.paths)):
      rwd += self.paths[i][2]
    returns = rwd/(self.config.batch_total*5)
    print('Current return is :', rwd/(self.config.batch_total*5))
    return returns



    
    


if __name__ == "__main__":
  env = gym.make('Hopper-v2')
  config = config_multi_sac.config_multi_sac(False)
  sa = sac(env, config)
  sa.train(2000)
  save_results(sa.loss_, 'results')
  #rendering(sa)

  
  