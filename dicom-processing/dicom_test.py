### Reading DICOM Files...
import os
from glob import glob
import scipy.misc
import numpy as np
import dicom


data_path = "../DWI"
#data_path = "./dicoms"
patients_dir = os.listdir(data_path)

output_path = "./"

#g = glob(patients_path+'/*.dcm') # 명시한 파일명을 스트링으로 불러오네.

g = glob(os.path.join(data_path, patients_dir[0], '*.dcm'))
print(g)
print(len(g))

print("Total of %d patient DICOM image. \nFirst 5 filenames:" %len(g))


print("Total of %d DICOM image. \nFirst 5 filenames:" %len(g))
print('\n'.join(g))



### Helper Functions ###
def load_scan(path):
    # Loop over the image files and store everything into a listd.
    """
    :param path: list of paths and its subdir will be refered
    :return: SliceThickness를 결정한 slices
    """
    import os
    import dicom
    import numpy as np

    slices = [dicom.read_file(path + '/' +s) for s in os.listdir(path) if "dcm" in s] #listdir은 dir을 받으면 안의 파일들의 리스트를 반환.
    slices.sort(key = lambda x : int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
    except: # what except? 아마 값이 없는 경우도 있나봄.
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # each dicom file object seem to have some attr such as ImagePositionPatient, SliceLocation, SliceThickness
    for s in slices:
        s.SliceThickness = slice_thickness #SliceThickness가 dicom객체의 attr로 존재하는 듯하다.
    return slices


def get_pixels_hu(scans):
    import numpy as np
    image = np.stack([s.pixel_array for s in scans]) # pixel_array는 dicom파일의 픽셀들을 볼 수 있다.

    # Convert to int16 (from sometimes int16),
    # Should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to HounsField units (HU)
    #intercept = scans[0].RescaleIntercept
    #slope = scans[0].RescaleSlope

    #if slope != 1:
        #image = slope* image.astype(np.float64)
        #image = image.astype(np.int16)
    #image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


#id=0
id=1
print(os.path.join(data_path, patients_dir[0]))
patient = load_scan(os.path.join(data_path, patients_dir[0]))
images = get_pixels_hu(patient)
import numpy as np

# it seems that after loading some info from scans, then store the result of some info including info of transformation from scans to img

#np.save(output_path+"fullimages_%d.npy" %(id), images)



### Displaying Images
#Lets now create a histogram of all the voxel data in the study.
import numpy as np
import matplotlib.pyplot as plt
id=1
file_used = output_path+"fullimages_%d.npy"%id
print(np.array(file_used).shape)
imgs_to_process = np.load(file_used).astype(np.float64)
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


### Critiquing the Histogram
'''
The histogram suggests the following:
There is lots of air
There is some lung
There's an abundance of soft tissue, mostly muscle, liver, etc, but there's also some fat.
There is only a small bit of bone(seen as a tiny sliver of height between 700-3000)
...
'''


### Displaying an Image Stack (Let's take a look at the actual images)
import numpy as np
import matplotlib.pyplot as plt
id = 0

imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
print(imgs_to_process.shape) # 481*512*512
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig, ax = plt.subplots(rows, cols, figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols), int(i%cols)].set_title('slice %d' % ind)
        ax[int(i/cols), int(i%cols)].imshow(stack[ind], cmap='gray')
        ax[int(i/cols), int(i%cols)].axis('off')
    plt.show()

def sample_plot(img, num_slice):
    plt.figure(figsize=[12,12])
    plt.imshow(img[num_slice], cmap='gray')
    plt.show()

sample_stack(imgs_to_process, rows=3, cols=5, start_with=0, show_every=1)
'''
for ind in range(100):
    print(ind)
    sample_plot(imgs_to_process, ind)
'''

