import os
import random
import numpy as np
import tensorflow as tf

from utils import *

class Conv3DNet():
    def __init__(self, session, i_depth, i_height, i_width, i_cdim, num_classes, num_data_size, batch_size, num_epoch, model_dir,
                 learning_rate=0.0001, train_rate=0.7, train_type="tvt", f_d=5, f_h=5, f_w=5, f_filter=32, beta1=0.5, forward_only=False):

        self.session = session
        self.i_depth = i_depth
        self.i_height = i_height
        self.i_width = i_width
        self.i_cdim = i_cdim
        self.num_classes = num_classes
        self.num_data_size = num_data_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.train_rate = train_rate
        self.train_type = train_type
        self.f_filter = f_filter
        self.beta1 = beta1
        self.forward_only = forward_only


    def create_model(self):
        # Placeholder

        self.X = tf.placeholder(tf.float32, [None, self.i_depth, self.i_height, self.i_width, self.i_cdim])
        self.Y = tf.placeholder(tf.int32, [None])
        self.Yhat = tf.one_hot(self.Y, self.num_classes)



        # batch normalization
        self.bn0 = batch_norm(name='bn0')
        self.bn1 = batch_norm(name='bn1')
        self.bn2 = batch_norm(name='bn2')

        self.logits, _ = self.layer(self.X, self.i_depth, self.i_height, self.i_width, self.f_filter)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(x=self.logits, y=self.Yhat))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Yhat, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables()) # saver object for created graphs
        self.load(self.model_dir)

    def load(self, model_dir):
        if os.path.isdir(model_dir) is False:
            os.makedirs(model_dir)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # check whether created checkpoint_path is
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else :
            print("Create model with new parameters.")
            self.session.run(tf.global_variables_initializer())

    def layer(self, X, f_depth, f_height, f_width, f_filter):
        h0 = lrelu(self.bn0(conv3d(X, self.f_filter, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_0')))
        h1 = lrelu(self.bn1(conv3d(h0, self.f_filter*2, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_1')))
        h2 = lrelu(self.bn2(conv3d(h1, self.f_filter*4, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='conv_3d_2')))
        shape = h2.get_shape().as_list()
        h2 = tf.reshape(h2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])
        h3 = linear(h2, self.num_classes, scope='h3_lin')

        return tf.nn.sigmoid(h3), h3

    def train(self, x, y):
        print("[*] Start Train Mode")

        print("x['tr_data']", x[0].shape)
        print("x['val_data']", x[1].shape)
        print("x['test_data']", x[2].shape)
        print("y['tr_label']", y[0].shape)
        print("y['val_label']", y[1].shape)
        print("y['test_label']", y[2].shape)

        train_batches_per_epoch = int(np.floor(len(x[0])/self.batch_size))

        for ep in range(self.num_epoch):
            for _ in range(train_batches_per_epoch):
                len_tr_data = len(x[0])
                batch_mask = np.random.choice(len_tr_data, self.batch_size)
                batch_xs = x[0][batch_mask]
                batch_ys = y[0][batch_mask]
                _, train_acc = self.session.run([self.optimizer,self.accuracy], feed_dict={self.X : batch_xs, self.Y : batch_ys})

            #if ep % 10 == 0:
                #s = sess.run(merged_summary, feed_dict={x: batch_xs,
                #                                        y: batch_ys,
                #                                        keep_prob: dropout_rate})
                #print("epoch ",ep,", Train Acc",train_acc)


            val_acc = self.session.run(self.accuracy, feed_dict={self.X:x[1], self.Y:y[1]})
            print("epoch ", ep, ", Validation Acc", val_acc, ", Train Acc", train_acc)

        test_acc = self.session.run(self.accuracy, feed_dict={self.X:x[2], self.Y:y[2]})
        print("Test Acc", test_acc)




        #self.optimizer

        return

    def test(self):
        print("[*] Start Test Mode")
        return


    def get_batch(self, datas): # num_classes, each_data_size, depth, height, width

        for ind in range(self.num_classes):
            random.shuffle(datas[ind])

        if self.train_type == "tvt":  # train() will treat test mode
            each_label = [[] for _ in range(3)]  # length of tr, val and test data

            tr_data=[]
            val_data=[]
            test_data=[]

            for data in datas:
                temp = data[:int(self.train_rate*len(data))]
                tr_data.append(temp[:int(0.9*len(temp))])
                val_data.append(temp[int(0.9*len(temp)):])
                test_data.append(data[int(self.train_rate*len(data)):])

                each_label[0].append(len(temp[:int(0.9*len(temp))]))
                each_label[1].append(len(temp[int(0.9*len(temp)):]))
                each_label[2].append(len(data[int(self.train_rate*len(data)):]))

            return np.vstack(np.array(tr_data)), np.vstack(np.array(val_data)), np.vstack(np.array(test_data)),\
        np.array([0]*each_label[0][0]+[1]*each_label[0][1]), \
                   np.array([0]*each_label[1][0]+[1]*each_label[1][1]), \
                   np.array([0]*each_label[2][0]+[1]*each_label[2][1])

        elif self.train_type == "tv":  # train() will treat only train and val
            each_label = [[] for _ in range(2)]  # length of tr and val data
            tr_data = []
            val_data = []
            for data in datas:
                tr_data.append(data[:int(self.train_rate*len(data))])
                val_data.append(data[int(self.train_rate*len(data)):])
                each_label[0].append(len(data[:int(self.train_rate*len(data))]))
                each_label[1].append(len(data[int(self.train_rate*len(data)):]))

            return np.vstack(np.array(tr_data)), np.vstack(np.array(val_data)), \
                   np.array([0] * each_label[0][0] + [1] * each_label[0][1]), \
                   np.array([0] * each_label[1][0] + [1] * each_label[1][1])





























