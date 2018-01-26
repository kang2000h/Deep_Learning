import dicom #read the dicom files
import os # do directory operations
import pandas as pd #nice for data analysis
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

data_dir = '/media/donga/Deep/Kaggle/stage1'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('/media/donga/Deep/Kaggle/stage1_labels.csv', index_col=0)

print(labels_df)

# Load files and check some dicom files
# for patient in patients[:10]:
#     label = labels_df.at[patient, 'cancer'] # labels_df.get_value(patient, 'cancer')
#     path = data_dir + patient # define the patient's path
#
#     slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
#     print(len(slices), slices[0].pixel_array.shape)

# And Plot them
# for patient in patients[:10]:
#     label = labels_df.at[patient, 'cancer'] # labels_df.get_value(patient, 'cancer')
#     path = data_dir + patient # define the patient's path
#
#     slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
#     plt.imshow(slices[0].pixel_array)
#     plt.show()

# Resize the Images
IMG_PX_SIZE = 256
#IMG_PX_SIZE = 100
HM_SLICES=20

def slice_it(li, num_of_chunks=2):
    start=0
    for i in range(num_of_chunks):
        stop = start + len(li[i::num_of_chunks]) # len(li[i::cols]) is the number remained from i to end on each cols step
        yield li[start:stop]
        start = stop

def chunks(l, n):
    # source: Ned Batchelder
    # Link:https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def mean(l):
    return sum(l)/len(l)

'''
for patient in patients[:10]:
    label = labels_df.at[patient, 'cancer'] # labels_df.get_value(patient, 'cancer')
    path = data_dir + patient # define the patient's path
    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    fig = plt.figure()
    for num, each_slice in enumerate(slices[:12]):
        y = fig.add_subplot(3, 4, num+1) # it will be [3, 4] sub plot grid
        new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE))
        plt.imshow(new_image)
    plt.show()


#make the length of depth
for patient in patients[:10]:
    label = labels_df.at[patient, 'cancer'] # labels_df.get_value(patient, 'cancer')
    path = data_dir + patient # define the patient's path
    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    # devide into chunks and average them respectively. then we will get the same size of matrices
    chunk_sizes = math.ceil(len(slices)/HM_SLICES)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk))) # need to be resolved by *
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES-1:
        new_slices.append(new_slices[-1])
    if len(new_slices) == HM_SLICES-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])
    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*(new_slices[HM_SLICES-1], new_slices[HM_SLICES]))))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1]=new_val
    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*(new_slices[HM_SLICES - 1], new_slices[HM_SLICES]))))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val


    print(len(new_slices))

    fig = plt.figure()
    for num, each_slice in enumerate(new_slices):
        y = fig.add_subplot(4, 5, num+1) # it will be [3, 4] sub plot grid
        y.imshow(each_slice)
    plt.show()
'''


# make it into function
def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.at[patient, 'cancer'] # labels_df.get_value(patient, 'cancer')
    path = os.path.join(data_dir, patient) # define the patient's path
    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []

    '''
        # devide into chunks and average them respectively. then we will get the same size of matrices
        chunk_sizes = math.ceil(len(slices)/HM_SLICES)

        # for slice_chunk in chunks(slices, chunk_sizes):
        #     slice_chunk = list(map(mean, zip(*slice_chunk))) # need to be resolved by *
        #     new_slices.append(slice_chunk)
        for slice_chunk in [*chunks(slices, chunk_sizes)]:
            new_slices.append(np.mean(slice_chunk,axis=0))

        if len(new_slices) == HM_SLICES-1:
            new_slices.append(new_slices[-1])
        if len(new_slices) == HM_SLICES-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
        if len(new_slices) == HM_SLICES + 2:
            new_val = list(map(mean, zip(*(new_slices[HM_SLICES-1], new_slices[HM_SLICES]))))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1]=new_val
        if len(new_slices) == HM_SLICES + 1:
            new_val = list(map(mean, zip(*(new_slices[HM_SLICES - 1], new_slices[HM_SLICES]))))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES - 1] = new_val
    '''

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    for slice_chunk in slice_it(slices, HM_SLICES):
        new_slices.append(np.mean(slice_chunk, axis=0))


    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num+1) # it will be [3, 4] sub plot grid
            y.imshow(each_slice)
        plt.show()

    if label==1: label=[0,1]
    elif label==0: label=[1,0]

    return new_slices, label # I changed for the objects to use only pythonic list

# we will save big list array into a file.
much_data = []

for num, patient in enumerate(patients):
    if num%100 == 0:
        print(num)
    try:
        img_data, label = process_data(patient, labels_df, img_px_size=IMG_PX_SIZE, hm_slices=HM_SLICES)
        much_data.append([img_data, label])
    except KeyError as e:
        print('This is unlabeled data')


# size of [numOfPatient,2], 2 means pixel value and labels
np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES), much_data)

print(len(much_data))








