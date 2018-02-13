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
import argparse

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help="Path of input data")
parser.add_argument('--checkpoint', type=str,help="Path of checkpoint")
parser.add_argument('--istrain', type=bool, default=False, help="define the mode of system")
parser.add_argument('--visualize', type=bool, default=False, help="show cnn visualization")
args = parser.parse_args()



def main():
    """
    Configuration Part.
    """

    # Path to the textfiles for the trainings and validation set
    train_file = './train.txt'
    val_file = './val.txt'

    # Learning params
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 10

    # Network params
    dropout_rate = 0.5
    num_classes = 2
    train_layers = ['fc8', 'fc7', 'fc6']

    # How often we want to write the tf.summary data to disk
    display_step = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = "./tmp/tensorboard"
    checkpoint_path = "./tmp/checkpoints"

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

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
    pred = tf.nn.softmax(score)

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
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
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
    train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    # Start Tensorflow session
    with tf.Session() as sess:
        print("# isTrain : ", args.istrain)
        if not args.istrain:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("checkpoint path isn't correct")
                return

            file = tf.read_file(args.input_path)
            decoded_img = tf.image.decode_jpeg(file, channels=3)
            resized_img = tf.image.resize_images(decoded_img, [227, 227])
            resized_img = tf.reshape(resized_img, [1, 227,227,3])
            print("# decoded img : ",decoded_img.eval().shape)


            pred_ = sess.run(pred, feed_dict={x: resized_img.eval(),
                                                y: [[1, 0]],
                                                keep_prob: np.array(1.0)})
            print("P(man|data)  : ", pred_[0][0])
            print("P(woman|data)  : ", pred_[0][1])

            img = decoded_img.eval()
            plt.imshow(img)
            plt.show()

            if args.visualize:
                w1, w2, w3, w4, w5 = sess.run(model.weight, feed_dict = {x: resized_img.eval(),
                                                y: [[1, 0]],
                                                keep_prob: np.array(1.0)})
                print("W1 : ", w1.shape)
                visualize(w1[:, :,0,:25])
                print("W2 : ", w2.shape)
                visualize(w2[:, :,0,:25])
                print("W3 : ", w3.shape)
                visualize(w3[:, :,0,:25])
                print("W4 : ", w4.shape)
                visualize(w4[:, :,0,:25])
                print("W5 : ", w5.shape)
                visualize(w5[:, :,0,:25])
                f1, f2, f3, f4, f5 = sess.run(model.fm, {x: resized_img.eval(),
                                                y: [[1, 0]],
                                                keep_prob: np.array(1.0)})
                print("F1 : ", f1.shape)
                visualize(f1[0][:,:,:25])
                print("F2 : ", f2.shape)
                visualize(f2[0][:, :, :25])
                print("F3 : ", f3.shape)
                visualize(f3[0][:, :, :25])
                print("F4 : ", f4.shape)
                visualize(f4[0][:, :, :25])
                print("F5 : ", f5.shape)
                visualize(f5[0][:, :, :25])

            return
        else:
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

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            train_acc = 0.
            train_count = 0.
            for step in range(train_batches_per_epoch):
                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)

                # And run the training op
                acc, _ = sess.run([accuracy, train_op], feed_dict={x: img_batch,
                                                                   y: label_batch,
                                                                   keep_prob: dropout_rate})
                train_acc += acc
                train_count += 1

                # Generate summary with the current batch of data and write to file
                # if step % display_step == 0:
                #     s = sess.run(merged_summary, feed_dict={x: img_batch,
                #                                             y: label_batch,
                #                                             keep_prob: 1.})
                #
                #     writer.add_summary(s, epoch*train_batches_per_epoch + step)
            train_acc /= train_count
            print("{} Train Accuracy = {:.4f}".format(datetime.now(),
                                                      train_acc))

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            sess.run(validation_init_op)
            test_acc = 0.
            test_count = 0

            for ind in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})
                test_acc += acc
                test_count += 1
                if epoch is 2 and ind is 0:
                    fm = sess.run(model.fm, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                    weight = sess.run(model.weight, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                    # print("fm0 : ", np.array(fm[0]).shape)
                    # print("fm1 : ", np.array(fm[1]).shape)
                    # print("fm2 : ", np.array(fm[2]).shape)
                    # print("fm3 : ", np.array(fm[3]).shape)
                    # print("fm4 : ", np.array(fm[4]).shape)
                    #
                    # print("weight0 : ", np.array(weight[0]).shape)
                    # print("weight1 : ", np.array(weight[1]).shape)
                    # print("weight2 : ", np.array(weight[2]).shape)
                    # print("weight3 : ", np.array(weight[3]).shape)
                    # print("weight4 : ", np.array(weight[4]).shape)

            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))


if __name__=="__main__":
    main()
