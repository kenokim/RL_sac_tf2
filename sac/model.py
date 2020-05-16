import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

class polinet(object):
  def __init__(self, env, in_dim, out_dim, continuity = True):
    self.env = env
    self.in_dim, self.out_dim = in_dim, out_dim
    self.continuity = continuity
    self.build()
  def build(self):
    inp = tf.keras.layers.Input(shape = (self.in_dim,))
    x = Dense(256, activation='relu')(inp)
    x = Dense(256, activation='relu')(x)
    if self.continuity :
      out1 = Dense(self.out_dim, activation = None)(x)
      out2 = Dense(self.out_dim, activation = None)(x)
      model = tf.keras.Model(inputs=inp, outputs=[out1, out2])
    else :
      out = Dense(self.out_dim, activation = 'softmax')(x)
      model = tf.keras.Model(inputs = inp, outputs = out)
    self.model = model
  def __call__(self, x):
    return self.model(x)



class valuenet(object):
  def __init__(self, env, epochs = 1):
    self.env = env
    self.build()
    self.x_train, self.y_train = None, None
    self.epochs = epochs
  def build(self):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
    opt = tf.keras.optimizers.Adam(0.003)
    self.opt = opt

    model.compile(loss = 'mse', optimizer = opt)#, metrics = ['mae', 'mse'])
    self.model = model
  def fit(self, x, y):
    self.model.fit(x, y, epochs = self.epochs, validation_split = 0.2, verbose = 0)
  def __call__(self, x):
    return self.model(x)
