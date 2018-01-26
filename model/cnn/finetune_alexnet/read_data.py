import glob
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

from glob import glob
import os

import tensorflow as tf

train_rate = 7
data_dir = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_paths(data_dir, extention, train_rate=7):
    num_total = len(glob(os.path.join(data_dir[0], extention)))

    tr_data = []
    tr_target = []
    val_data = []
    val_target = []

    for ind, class_ in enumerate(data_dir):    
        tmp = glob(os.path.join(class_, "*.png"))                
        num_size = len(tmp)
        num_train = int(num_size*train_rate/10)
        tr_class = np.array([ind]*num_train)
        val_class = np.array([ind]*(num_total-num_train))
	
        tr_data.append(tmp[:num_train])
        tr_target.append(tr_class)
        val_data.append(tmp[num_train:])
        val_target.append(val_class)
    return (tr_data, tr_target, val_data, val_target, num_total, num_train)
    

def read_img(paths, size):
    tr_img = []
    for class_ in paths:
        tmp = []
        for img in class_:
            pil_obj = PIL.Image.open(img) 
            pil_obj = pil_obj.resize(size, Image.ANTIALIAS)
            tmp.append(np.array(pil_obj))
        tr_img.append(tmp)
    tr_img = np.array(tr_img)
    return tr_img

def get_path_v2(filename):
    paths = []
    labels = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(" ")
            paths.append(line[0])
            labels.append(int(line[1][0]))
    return paths, labels


def read_img_v2(paths, size):    
    imgs = []
    for path in paths:
        pil_obj = PIL.Image.open(path)
        pil_obj = pil_obj.resize(size, Image.ANTIALIAS)

        imgs.append(np.array(pil_obj))

    imgs = np.array(imgs)
    return imgs

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
def read_img_v3(paths, size):    
    imgs = []

    for path in paths:        
#        pil_obj = PIL.Image.open(path)
#        pil_obj = pil_obj.resize(size, Image.ANTIALIAS)
        img_decoded = tf.image.decode_png(path, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [size[0], size[1]])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
#        imgs.append(np.array(pil_obj))

#    imgs = np.array(imgs)
    return img_bgr

def read_dicom_img(paths, size):
    return


#tr_data, val_data, _, _,  _ = get_paths(data_dir, "*.png")
#tr_img = read_img(tr_data, (28, 28))
#print(tr_img)
#print(tr_img.shape)
#print(len(tr_img))







