# Dicom Prosessing
This modules are created to process Dicom images when processing or analyzing medical images.

## Role & Dependencies of each modules
-  dicom_loader.py (depends on 'pydicom')

Dicom files have many properties and designed for entire medical work.
huge dicom files about even 64GB are sparse to us to doing image processing.
This module convert the entire dicom files about the patients in all data to numpy files that have only pixel_array property of all dicom files and you can read the numpy array files for your model

### Usage for dicom_loader.py

The first time a dicom files have been loaded

`python dicom_loader.py --data_dir /home/donga/neurology/data --class_list CE LAA --img_px_size 50 --slices 20 --save_filename ./data/neurology-50-50-20.npy`

There is a file already created in 'save_filename'

`python dicom_loader.py --save_filename ./data/muchdata-50-50-20.npy`


-  dicom_viewer.py (depends on 'pydicom', 'glob', 'numpy', 'matplot.pyplot')
dicom files are stored as a *.dcm type, so we need to use dicom processing modules
I just made simple module for viewing dicom files for each patient

### Usage for dicom_viewer.py

`python dicom_viewer.py --data_path data/CE --output_path ./dicom_viewer --save_filename neurology.npy`
`python dicom_viewer.py --data_path data/LAA --output_path ./dicom_viewer --save_filename neurology.npy --visualize True`

### Usage for dicom_viewer.py 

`python dicom_viewer.py --data_path data/CE --output_path ./dicom_viewer --save_filename neurology.npy`

### Usage for dicom_viewer.py to call these module on a script

```
# converting our fbb dicom data into voxels
import os
import glob
import numpy as np

from dicom_viewer import *

filename ='/media/donga/Deep/entire_pet'
classes = os.listdir(filename)
patient = os.listdir(os.path.join(filename, classes[0]))
milestone = 'fbbstatic'

slices = load_scan(os.path.join(filename, classes[0], patient[0], milestone))
instance = get_pixels_hu(slices)

viewer(instance, rows=5, cols=6, show_every=3) 
#viewer(instance, rows=10, cols=11, show_every=1) # show all slices
```






