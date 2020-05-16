from actor import *
from utils import *



def train_critic2(batch, model, qnet, qnet2, tarq, tarq2, alpha = 1, gamma = 0.99, action_constant1 = 1, action_constant2 = 0):
  obs, act, rwd, nobs, dones = path_reshape(batch)
  q_in = tf.concat([obs, act], axis = -1)
  nmu, nlog_std = model(tf.cast(nobs, tf.float32))
  nlog_std = tf.clip_by_value(nlog_std, -20, 2)
  nstd = tf.math.exp(nlog_std)
  action, log_pi = action_log_pi(nmu, nstd, action_constant1, action_constant2)
  nq_in = tf.concat([nobs, action], axis = -1)
  nqval = tf.math.minimum(tarq(nq_in), tarq2(nq_in))
  masking = tf.ones((np.shape(dones))) - dones
  masking = tf.expand_dims(masking, axis = -1)
  qval = tf.multiply(nqval, masking)
  rwd = tf.expand_dims(rwd, axis = -1)
  q_tar = rwd + gamma * qval
  qnet.fit(q_in, q_tar)
  qnet2.fit(q_in, q_tar)
  return None


def train_value(batch, model, qnet, qnet2, tarq, tarq2, alpha = 1, gamma = 0.99, action_constant1 = 1, action_constant2 = 0):
  obs, act, rwd, nobs, dones = path_reshape(batch)
  q_in = tf.concat([obs, act], axis = -1)
  nmu, nlog_std = model(nobs)
  nlog_std = tf.clip_by_value(nlog_std, -20, 2)
  nstd = tf.math.exp(nlog_std)
  action, log_pi = action_log_pi(nmu, nstd, action_constant1, action_constant2)
  nq_in = tf.concat([nobs, action], axis = -1)
  nqval = tf.math.minimum(tarq(nq_in), tarq2(nq_in))
  masking = tf.ones((np.shape(dones))) - dones
  masking = tf.expand_dims(masking, axis = -1)
  log_pi = tf.reduce_sum(log_pi, axis = -1)
  log_pi = tf.expand_dims(log_pi, axis = -1)
  nqval = nqval - log_pi
  qval = tf.multiply(nqval, masking)
  rwd = tf.expand_dims(rwd, axis = -1)
  q_tar = rwd + gamma * qval
  qnet.fit(q_in, q_tar)
  qnet2.fit(q_in, q_tar)


def policy_train2():
  @tf.function()
  def policy_training2(obs, model, qnet, opts, noise, log_prob):
    with tf.GradientTape() as tape:
      mu, log_std = model(obs)
      std = tf.math.exp(log_std)
      action = tf.multiply(noise, std)
      action = tf.math.add(action, mu)
      action = tf.math.tanh(action)
      tanh_act = tf.math.square(action)
      log_pi = log_prob - tf.reduce_sum(tf.math.log((1-tanh_act)), axis = -1)
      q_in = tf.concat([obs, action], axis = -1)
      q_tar = qnet(q_in)
      log_pi = tf.expand_dims(log_pi, axis = -1)
      loss_ = tf.reduce_mean(log_pi - q_tar)
    grad = tape.gradient(loss_, model.trainable_weights)
    opts.apply_gradients(zip(grad, model.trainable_weights))
    return loss_, tf.reduce_mean(q_tar), -tf.reduce_mean(log_pi)
  return policy_training2

def policy_train2():
  @tf.function()
  def policy_training2(obs, model, qnet, opts, noise, log_prob):
    with tf.GradientTape() as tape:
      mu, log_std = model(obs)
      std = tf.math.exp(log_std)
      #action, log_pi = action_log_pi(mu, std)
      action = tf.multiply(noise, std)
      action = tf.math.add(action, mu)
      action = tf.math.tanh(action)
      tanh_act = tf.math.square(action)
      log_pi = log_prob - tf.reduce_sum(tf.math.log((1-tanh_act)), axis = -1)
      q_in = tf.concat([obs, action], axis = -1)
      q_tar = qnet(q_in)
      log_pi = tf.expand_dims(log_pi, axis = -1)
      loss_ = tf.reduce_mean(log_pi - q_tar)
    grad = tape.gradient(loss_, model.trainable_weights)
    opts.apply_gradients(zip(grad, model.trainable_weights))
    return loss_, tf.reduce_mean(q_tar), -tf.reduce_mean(log_pi)
  return policy_training




def policy_training():
  @tf.function()
  def policy_training(obs, model, qnet, opts, noise):
    with tf.GradientTape() as tape:
      mu, log_std = model(obs)
      std = tf.math.exp(log_std)
      action = tf.multiply(std, noise) + mu
      #action = tf.math.add(action, mu)
      log_pi = normal_pdf(action, mu, std)
      action = tf.math.tanh(action)
      tanh_act = tf.math.square(action)
      log_pi = log_pi - tf.math.log(1-tanh_act)
      q_in = tf.concat([obs, action], axis = -1)
      q_tar = qnet(q_in)
      log_pi = tf.reduce_sum(log_pi, axis = -1)
      log_pi = tf.expand_dims(log_pi, axis = -1)
      loss_ = tf.reduce_mean(log_pi - q_tar)
    grad = tape.gradient(loss_, model.trainable_weights)
    opts.apply_gradients(zip(grad, model.trainable_weights))
    return loss_, tf.reduce_mean(q_tar), -tf.reduce_mean(log_pi)
  return policy_training