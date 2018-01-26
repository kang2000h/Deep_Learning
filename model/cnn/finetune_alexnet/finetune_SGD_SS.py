"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os
import sys

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from read_data import *
import random

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
#train_file = '/path/to/train.txt'
train_file = 'train.txt'
#val_file = '/path/to/val.txt'
val_file = 'val.txt'

OPTION = sys.argv

if "--train_file" in OPTION:
  train_file_op_idx = OPTION.index("--train_file")+1
  train_file = OPTION[train_file_op_idx]

if "--val_file" in OPTION:
  val_file_op_idx = OPTION.index("--val_file")+1
  val_file = OPTION[val_file_op_idx]

cv_file = None
if "--cv_file" in OPTION:
  cv_file_op_idx = OPTION.index("--cv_file")+1
  cv_file = OPTION[cv_file_op_idx]

useDG = False
if "--isDG" in OPTION:
    useDG_idx = OPTION.index("--isDG") +1
    useDG = bool(OPTION[useDG_idx])

ss_file = None
num_val_hc = 0
num_val_mci = 0
num_val_ad = 0
if "--ss_file" in OPTION:
  ss_file_op_idx = OPTION.index("--ss_file")+1
  ss_file = OPTION[ss_file_op_idx]
  num_val_hc = int(OPTION[OPTION.index("--ss_file")+2])
  num_val_mci = int(OPTION[OPTION.index("--ss_file") + 3])
  num_val_ad = int(OPTION[OPTION.index("--ss_file") + 4])

dg_file = None
if "--df_file" in OPTION:
    dg_file_op_idx = OPTION.index("--dg_file")+1
    dg_file = OPTION[dg_file_op_idx]

if "--plot_filename" in OPTION:
  plot_data_op_idx = OPTION.index("--plot_filename")+1
  plot_filename = OPTION[plot_data_op_idx]

# Learning params
learning_rate = 0.00005
#num_epochs = 100
num_epochs = 1
#batch_size = 128
batch_size=1

# Network params
dropout_rate = 0.5
#num_classes = 2
num_classes=3
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
#display_step = 20
display_step = 10

# Path for tf.summary.FileWriter and to store model checkpoints
#filewriter_path = "/tmp/finetune_alexnet/tensorboard"
filewriter_path = "tmp/finetune_alexnet/tensorboard"
#checkpoint_path = "/tmp/finetune_alexnet/checkpoints"
checkpoint_path = "tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    #os.mkdir(checkpoint_path)
    os.makedirs(checkpoint_path)

'''
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
'''

# Ops for initializing the two different iterators
#training_init_op = iterator.make_initializer(tr_data.data)
#validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

def sensitive_loss(loss, score, y):
    '''
    :param loss : batch_size,
    :param score: batch_size * num_class
    :param y: batch_size * num_class
    :return: size of batch_size, to be doubled to a wrong elements
    '''
    hit_true = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    hit_true = tf.cast(hit_true, tf.float32)
    hit_false = tf.less(hit_true, 1)

    mci = tf.equal(tf.argmax(y, 1) , 1)
    sensitive_loss_mask = hit_false & mci
    sensitive_loss = (tf.cast(sensitive_loss_mask, tf.float32) *2 ) +1
    return tf.multiply(loss,  sensitive_loss)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y)
    losses = sensitive_loss(losses, score, y)
    loss = tf.reduce_mean(losses)

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

def zero_centered(inputs):
    MEAN = np.mean(np.mean(inputs, axis=0).astype(np.float32), axis=0).astype(np.float32)
    return inputs-MEAN

if cv_file is not None:
    # get entire data
    all_data_paths, val_labels = get_path_v2(cv_file)
    all_imgs = read_img_v2(all_data_paths, (227, 227))
    all_imgs = np.reshape(all_imgs, (-1, 227, 227, 3))
    all_imgs = zero_centered(all_imgs)
    val_labels_onehot = np.eye(num_classes)[val_labels]

    from sklearn.model_selection import KFold, cross_val_score

    num_K = 4
    total_test_Acc = 0
    total_conf_matrix = []
    k_fold = KFold(n_splits=num_K)
    count = 1
    for train_indices, test_indices in k_fold.split(all_imgs, val_labels_onehot):
        tr_imgs = all_imgs[train_indices, :, :, :]
        tr_labels_onehot = val_labels_onehot[train_indices, :]
        val_imgs = all_imgs[test_indices, :, :, :]
        val_labels_onehot = val_labels_onehot[test_indices, :]
        break
elif ss_file is not None and useDG is True: # adopting stratified sampling while doing Data Augmentation
    # get entire real data
    all_data_paths, val_labels = get_path_v2(ss_file)

    # divide into classes respectively
    divided_all_realdata_paths = [[] for _ in range(num_classes)]
    for ind, label in enumerate(val_labels):
        divided_all_realdata_paths[label].append(all_data_paths[ind])
    random.shuffle(divided_all_realdata_paths[0])
    random.shuffle(divided_all_realdata_paths[1])
    random.shuffle(divided_all_realdata_paths[2])

    val_paths = divided_all_realdata_paths[0][:num_val_hc] + divided_all_realdata_paths[1][:num_val_mci] + \
                divided_all_realdata_paths[2][:num_val_ad]
    val_labels = [0] * num_val_hc + [1] * num_val_mci + [2] * num_val_ad

    val_imgs = read_img_v2(val_paths, (227, 227))
    val_imgs = np.reshape(val_imgs, (-1, 227, 227, 3))
    val_imgs = zero_centered(val_imgs)
    val_labels_onehot = np.eye(num_classes)[val_labels]

    tr_paths = divided_all_realdata_paths[0][num_val_hc:] + divided_all_realdata_paths[1][num_val_mci:] + \
               divided_all_realdata_paths[2][num_val_ad:]
    tr_labels = [0] * (len(divided_all_realdata_paths[0]) - num_val_hc) + [1] * (
    len(divided_all_realdata_paths[1]) - num_val_mci) + [2] * (len(divided_all_realdata_paths[2]) - num_val_ad)

    all_gan_paths, gan_val_labels= get_path_v2(dg_file)
    tr_paths += all_gan_paths
    tr_labels += gan_val_labels

    tr_imgs = read_img_v2(tr_paths, (227, 227))
    tr_imgs = np.reshape(tr_imgs, (-1, 227, 227, 3))
    tr_imgs = zero_centered(tr_imgs)
    tr_labels_onehot = np.eye(num_classes)[tr_labels]

elif ss_file is not None: # adopting stratified sampling
    # get entire data
    all_data_paths, val_labels = get_path_v2(ss_file)
    # divide into classes respectively
    divided_all_data_paths = [[] for _ in range(num_classes)]
    for ind, label in enumerate(val_labels):
        divided_all_data_paths[label].append(all_data_paths[ind])
    random.shuffle(divided_all_data_paths[0])
    random.shuffle(divided_all_data_paths[1])
    random.shuffle(divided_all_data_paths[2])


    val_paths = divided_all_data_paths[0][:num_val_hc]+ divided_all_data_paths[1][:num_val_mci]+ divided_all_data_paths[2][:num_val_ad]
    val_labels = [0]*num_val_hc+[1]*num_val_mci+[2]*num_val_ad

    tr_paths = divided_all_data_paths[0][num_val_hc:]+divided_all_data_paths[1][num_val_mci:]+divided_all_data_paths[2][num_val_ad:]
    tr_labels = [0] * (len(divided_all_data_paths[0])-num_val_hc) + [1] *(len(divided_all_data_paths[1])-num_val_mci) + [2] * (len(divided_all_data_paths[2])-num_val_ad)


    tr_imgs = read_img_v2(tr_paths, (227, 227))
    tr_imgs = np.reshape(tr_imgs, (-1, 227, 227, 3))
    tr_imgs = zero_centered(tr_imgs)
    tr_labels_onehot = np.eye(num_classes)[tr_labels]

    val_imgs = read_img_v2(val_paths, (227, 227))
    val_imgs = np.reshape(val_imgs, (-1, 227, 227, 3))
    val_imgs = zero_centered(val_imgs)
    val_labels_onehot = np.eye(num_classes)[val_labels]

else :
    tr_paths, tr_labels = get_path_v2(train_file)
    tr_imgs = read_img_v2(tr_paths, (227, 227))
    tr_imgs = np.reshape(tr_imgs, (-1, 227, 227, 3))
    tr_imgs = zero_centered(tr_imgs)
    tr_labels_onehot = np.eye(num_classes)[tr_labels]

    val_paths, val_labels = get_path_v2(val_file)
    val_imgs = read_img_v2(val_paths, (227, 227))
    val_imgs = np.reshape(val_imgs, (-1, 227, 227, 3))
    val_imgs = zero_centered(val_imgs)
    val_labels_onehot = np.eye(num_classes)[val_labels]

# Get the number of training/validation steps per epoch

train_batches_per_epoch = int(np.floor(len(tr_imgs) / batch_size))
val_batches_per_epoch = int(np.floor(len(val_imgs) / batch_size))


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True


# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    ta = []
    va = []

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

   # Loop over number of epochs
    for epoch in range(num_epochs):

        train_acc = 0.
        train_count = 0

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        #sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            batch_mask = np.random.choice(len(tr_imgs), batch_size)
            batch_xs = tr_imgs[batch_mask]
            print(batch_xs.shape)
            batch_ys = tr_labels_onehot[batch_mask]
            print(batch_xs[0])
            # And run the training op
            y_, score_, tacc, _, loss_ = sess.run([y, score, accuracy, train_op, loss], feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: dropout_rate})
                print("step:",step, "loss",loss_)


            train_acc+= tacc
            train_count +=1
        train_acc /= train_count
        print("{} Training Accuracy = {:.4f}".format(datetime.now(),
                                                       train_acc))

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        #sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        conf_matrix = np.zeros([num_classes,num_classes])

        for i in range(val_batches_per_epoch):

            batch_xs = val_imgs[i*batch_size:i*batch_size+batch_size]
            batch_ys = val_labels_onehot[i*batch_size:i*batch_size+batch_size]
            acc, score_, y_ = sess.run([accuracy, score, y], feed_dict={x: batch_xs,
                                                y: batch_ys,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            for idx in range(len(batch_xs)):
                conf_matrix[np.argmax(y_[idx])][np.argmax(score_[idx])]+=1
        test_acc /= test_count
        print("Confusion Matrix")
        print(conf_matrix)
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))

        ta.append(train_acc)
        va.append(test_acc)

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams.update({'font.size':22})

matplotlib.rc('font', size=23)
matplotlib.rc('axes', labelsize=25)
matplotlib.rc('legend', fontsize=25)

x = np.arange(len(ta))
y1 = np.array(ta)*100
y2 = np.array(va)*100

plt.plot(x, y1, label="Train")
plt.plot(x, y2, linestyle="--", label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.legend()


plt.show()


with open(plot_filename, "w") as f:
    count=1
    f.write("Epoch"+"\t"+"Train_Acc"+"\t"+"Val_Acc"+"\n")
    for t, v in zip(ta, va):
        f.write(str(count)+"\t"+str(t)+"\t"+str(v)+"\n")
        count+=1
