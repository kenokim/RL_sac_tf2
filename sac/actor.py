import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def sample_path(env, model, batch_size, max_ep_len, continuity, num_episodes = None):
    episode = 0
    paths = []
    t = 0
    while (num_episodes or t < batch_size):
      state = env.reset()
      obs_len = len(state)
      for step in range(max_ep_len):
        box = []
        box.append(state)
        state = tf.cast(tf.expand_dims(state, axis = 0), tf.float32)
        if continuity :
          mu_logsig = model(state)
          mu, log_sig = mu_logsig[0], mu_logsig[1]
          log_sig = tf.clip_by_value(log_sig, -20, 2)
          sig = tf.math.exp(log_sig)
          action, _ = action_log_pi(mu, sig)
          action = np.concatenate(action)
        else:
          mean = model(state)
          action = np.random.choice(obs_len, p = mean[0].numpy())
        state, reward, done, info = env.step(action)
        box.append(action)
        box.append(reward * 5)
        box.append(state)
        box.append(done)
        t += 1
        if (done or step == max_ep_len-1):
          break          
        if (not num_episodes) and t == batch_size:
          break
        paths.append(box)        
      episode += 1
      if num_episodes and episode >= num_episodes:
        break
      paths.append(box)
    return paths



def action_log_pi2(mean, std, const1 = 1, const2 = 0):
  dist = tfd.MultivariateNormalDiag(loc = np.zeros(np.shape(mean)), scale_diag = np.ones(np.shape(std)))
  noise = dist.sample()
  log_pi = tf.math.log(dist.prob(noise))
  noise = tf.cast(noise, tf.float32)
  action = tf.multiply(noise, std) + mean
  action = tf.math.tanh(action) * const1 - const2
  tanh_action = tf.math.square(action)
  cor = tf.reduce_sum(tf.math.log((1-tanh_action)), axis = -1)
  log_pi = tf.cast(log_pi, tf.float32)
  log_pi = log_pi - cor
  log_pi = tf.expand_dims(log_pi, axis = -1)
  return action, log_pi


def action_log_pi2(mean, std, const1 = 1, const2 = 0):
  dist = tfd.MultivariateNormalDiag(loc = mean, scale_diag = std)
  noise = dist.sample()
  noise = tf.stop_gradient(noise)
  log_pi = tf.math.log(dist.prob(noise))
  action = tf.cast(noise, tf.float32)
  action = tf.stop_gradient(action)
  action = tf.math.tanh(action) * const1 - const2
  tanh_action = tf.math.square(action)
  cor = tf.reduce_sum(tf.math.log((1-tanh_action)), axis = -1)
  log_pi = tf.cast(log_pi, tf.float32)
  log_pi = log_pi - cor
  log_pi = tf.expand_dims(log_pi, axis = -1)
  return action, log_pi

def action_log_pi(mu, sig, const1 = 1, const2 = 0):
  shape = np.shape(mu)
  noise = sampler(shape[0], shape[1])
  action = noise * sig + mu
  log_pi = normal_pdf(action, mu, sig)
  action = tf.math.tanh(action)
  tanh_action = tf.math.square(action)
  cor = tf.math.log(1-tanh_action)
  log_pi = tf.cast(log_pi, tf.float32)
  log_pi = log_pi - cor
  return action, log_pi

def sampler(m, dim):
    sample = np.random.normal(0, 1, size = [m, dim])
    sample = tf.cast(sample, tf.float32)
    return sample

def normal_pdf(x, mu, sig):
    x = tf.cast(x, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sig = tf.cast(sig, tf.float32)
    g = 1/(sig * tf.math.sqrt(2 * np.pi))
    p = tf.math.exp(-0.5 * ((x - mu) / sig) ** 2)
    return g * p

def factorize(noise, mu, sig):
    return noise * sig + mu


def sample_():
    eps = tf.random.normal(shape = None)
@tf.function()    
def normal_prob(x, mu, sig):
    #pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
    #Z = (2 pi sigma**2)**0.5
    Z = (2 * np.pi * (sig ** 2)) ** 0.5
    val = tf.math.exp(-0.5 * ((x - mu) ** 2) / (sig ** 2))
    return val/Z
@tf.function()
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


#x = np.array([1,2])
#mean = np.array([0,1])

#sig = np.array([1,1])
#var = sig ** 2
#logvar = tf.math.log(var)

#v1 = log_normal_pdf(x, mean, sig)
#v2 = normal_prob(x, mean, sig)