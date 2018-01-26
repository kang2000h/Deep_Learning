import pprint
import tensorflow as tf

pp = pprint.PrettyPrinter()

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def conv3d(x, output_channel, k_d=5, k_h=5, k_w=5, s_d=2, s_h=2, s_w=2, stddev=0.02, name="conv3d"):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', [k_d, k_h, k_w, x.get_shape()[-1], output_channel],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(x, w, strides=[1, s_d, s_h, s_w, 1], padding='VALID')
        biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

    return conv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                             tf.contrib.layers.xavier_initializer(False))
        bias = tf.get_variable("bias", [output_size], initializer=tf.contrib.layers.xavier_initializer(False))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else :
            return tf.matmul(input_, matrix) + bias


# we need to define the stride of depth and size of filter compared to 2d Conv
def maxpool3d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

'''
import numpy as np
much_data = np.load('./data/muchdata-50-50-20.npy') # 1397, 2 -> num_classes, num_data, depth, height, width, channel=1

# print(much_data.shape)
# print(np.array(much_data[0][0]).shape)
# print(much_data[0][1])

data = np.array([cont[0] for cont in much_data])
label = np.array([cont[1] for cont in much_data])

#data = np.expand_dims(data, axis=len(data.shape))
print(data.shape)


label = np.array([np.argmax(cont) for cont in label])
print(label.shape)

tmp = [[] for _ in range(2)]
for ind, val in enumerate(label):
    tmp[val].append(np.array(data[ind]))

tmp = np.array(tmp)
print(tmp.shape)
print(np.array(tmp[0]).shape)
print(np.array(tmp[1]).shape)
np.save('./data/kaggle_data.npy',tmp)
'''


