import tensorflow as tf
import matplotlib.pyplot as plt

def path_reshape(paths):
  obs, act, rwd, nobs, dones = [], [], [], [], []
  for i in range(len(paths)):
    obs.append(paths[i][0])
    act.append(paths[i][1])
    rwd.append(paths[i][2])
    nobs.append(paths[i][3])
    dones.append(paths[i][4])
  obs = tf.cast(obs, tf.float32)
  nobs = tf.cast(nobs, tf.float32)
  rwd = tf.cast(rwd, tf.float32)
  return obs, act, rwd, nobs, dones


def update_network_weights(rate, net, target):
  v_hat = [x * rate for x in net.get_weights()]
  v = [x * (1-rate) for x in target.get_weights()]
  weights = [sum(x) for x in zip(v_hat, v)]
  target.set_weights(weights)
  
def rendering(obj):
    env = obj.env
    obs = env.reset()
    for _ in range(10000):
        env.render()
        #action = obj.sample_action(tf.cast(obs, tf.float32))
        action, _ = obj.model(tf.cast(tf.expand_dims(obs, axis = 0), tf.float32))
        obs, rwd, done, _ = env.step(action)
        if done:
            obs = env.reset()
    env.close()
    

def save_results(val, file_name):
    assert type(val) == list 
    txt = file_name + '.txt'
    try:
        new_vals = []
        with open(txt, 'r') as g:
            vals = g.read().split('\n')[:-1]
            for i in range(int(len(vals)/2)):
                new_vals.append([vals[2*i], vals[2*i+1]])
    except:
        pass
                
    with open(txt, 'w') as f:
        try:
            val = new_vals + val
        except:
            pass
        for item in val:
            for k in item:
                f.write("%s\n" % k)
                
                
def graph(val):
    plt.plot(val)
    
    
    
    
    
             
