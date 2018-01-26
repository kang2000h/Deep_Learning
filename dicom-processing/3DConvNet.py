import tensorflow as tf
import numpy as np

IMG_PX_SIZE = 50
HM_SLICES = 20

n_classes=2
batch_size=100

x = tf.placeholder(tf.float32, [None, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE])
y = tf.placeholder(tf.int16, [None, n_classes])

dropout_rate = 0.5
keep_prob = tf.placeholder(tf.float32)

# we need to define the stride of depth compared to 2d Conv
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

# we need to define the stride of depth and size of filter compared to 2d Conv
def maxpool3d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    # we need to define the 5d filters for 3d conv
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])), #1 channel, 32 feature
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 5, 32, 64])), #32 channel, 64 feature
               'W_fc': tf.Variable(tf.random_normal([5*13*13*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 1]) # channel of resource data is 1

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1,5*13*13*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, dropout_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    #much_data = np.load('muchdata-256-256-20.npy') # None,2
    much_data = np.load('muchdata-50-50-20.npy')  # None,2

    print("much_data",much_data.shape)
    train_data = much_data[:-300]
    print("##############",len(train_data))
    validation_data = much_data[-300:]
    print("##############",len(validation_data))
    print("train_data", train_data.shape)
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    hm_epochs = 50
    iter_per_epochs = int(np.floor(len(train_data)/batch_size))
    with tf.Session(config = tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            for iter in range(iter_per_epochs):
                epoch_loss = 0
                batch_mask = np.random.choice(len(train_data), batch_size)
                batch_data = train_data[batch_mask]
                try:
                    batch_xs = np.stack(batch_data[:,0],axis=0) # np.stack() make list of numpy array into numpy matrice
                    batch_ys = np.stack(batch_data[:,1],axis=0)
                except ValueError as ve:
                    print(ve)
                    continue

                _, c,acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob:dropout_rate})

                epoch_loss += c
#                print("ISGPU",isgpu_)
                print('BatchNum', iter+1, 'completed out of', hm_epochs, 'loss:', epoch_loss, 'Accuracy:', acc)
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss, 'Accuracy:',acc)


        batch_xs = np.stack(validation_data[:,0],axis=0)
        batch_ys = np.stack(validation_data[:,1],axis=0)

        print('Accuracy:', accuracy.eval({x: batch_xs, y: batch_ys, keep_prob:1.}))


train_neural_network(x)