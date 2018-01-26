import dicom #read the dicom files
import os # do directory operations
import argparse
import pandas as pd #nice for data analysis
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math



def slice_it(li, num_of_chunks=2):
    start=0
    for i in range(num_of_chunks):
        stop = start + len(li[i::num_of_chunks]) # len(li[i::cols]) is the number remained from i to end on each cols step
        yield li[start:stop]
        start = stop

def mean(l):
    return sum(l)/len(l)

#
def process_data(each_patient, IMG_PX_SIZE, HM_SLICES):
    '''
    :param patient: path of each patient_dir
    :param visualize: whether figures are needed
    :return:
    '''

    slices = [dicom.read_file(each_patient+'/'+s) for s in os.listdir(each_patient)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    for slice_chunk in slice_it(slices, HM_SLICES):
        new_slices.append(np.mean(slice_chunk,axis=0))

    return np.array(new_slices) # I changed for the objects to use only pythonic list


def dicom_loader(save_filename, data_dir, class_list, img_h_w, img_d):
    if os.path.isfile(save_filename) is True:
        print("that filename is already stored!")
    else:
        # we will save big list array into a file.
        much_data = []  # list of class data

        for num, each_class in enumerate(class_list):
            each_class_data = []
            for num, each_patient in enumerate(os.listdir(os.path.join(data_dir, each_class))):
                if num % 100 == 0:
                    print(num)
                try:
                    img_data = process_data(os.path.join(data_dir, each_class, each_patient), img_h_w, img_d)
                    each_class_data.append(img_data)
                except KeyError as e:
                    print('This is unlabeled data')
                    pass
                except NotADirectoryError as nade:
                    print(nade)
                    pass
            much_data.append(each_class_data)
        np.save(save_filename, np.array(much_data))
    much_data = np.load(save_filename)
    return much_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dicom directory you need to process")
    parser.add_argument('--class_list', type=str, nargs='+',help="list of class")
    parser.add_argument('--save_filename', type=str,
                        help="filename of numpy array that have dicoms pixels")
    parser.add_argument('--img_px_size', type=int, help="img size to resize")
    parser.add_argument('--slices', type=int, help="number of slices to norm")
    IMG_PX_SIZE = 256  # required
    HM_SLICES = 20  # required

    args = parser.parse_args()


    test = dicom_loader(args.save_filename)
    print(test.shape)

if __name__=="__main__":
    main()


