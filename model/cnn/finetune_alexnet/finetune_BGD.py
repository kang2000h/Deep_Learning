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

# Learning params
learning_rate = 0.0001
num_epochs = 25
#batch_size = 128
batch_size=20

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

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

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

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

tr_paths, tr_labels = get_path_v2(train_file)
tr_imgs = read_img_v2(tr_paths, (227, 227))
tr_imgs = np.reshape(tr_imgs, (-1, 227, 227, 3))
tr_labels_onehot = np.eye(num_classes)[tr_labels]

val_paths, val_labels = get_path_v2(val_file)
val_imgs = read_img_v2(val_paths, (227, 227))
val_imgs = np.reshape(val_imgs, (-1, 227, 227, 3))
val_labels_onehot = np.eye(num_classes)[val_labels]

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        
        # And run the training op
        _, loss_ = sess.run([train_op, loss], feed_dict={x: tr_imgs,
                                          y: tr_labels_onehot,
                                          keep_prob: dropout_rate})
        print("Epoch : ", epoch, "Loss : ", loss_)                   


        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        conf_matrix = np.zeros([num_classes,num_classes])

        acc, score_, y_ = sess.run([accuracy, score, y], feed_dict={x: val_imgs,
                                                y: val_labels_onehot,
                                                keep_prob: 1.})
        print("score_", score_.shape)
        print("y_", y_.shape)
        print("np.argmax(score)", score_[0])
        print("np.argmax(y)", y_[0])
        test_acc += acc
        test_count += 1
        for idx in range(len(val_imgs)):
            conf_matrix[np.argmax(score_[idx])][np.argmax(y_[idx])]+=1
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
