
# Data Description

The training data (x_train) for this project consists of an hdf5 file that contain the image volumes, and a csv file that contains the metadata. We provide a `read.py` file that reads this for you to get you going quickly.

The X hdf5 file contains 1400 64x64x64 scans (zero indexed). The final output of the approach should be the true positions (with or without model post processing). You can predict the diameters if you'd like but this isnt necessary for the challenge as in each image all the particles have the same size.

The metadata parameters are as follows:
* volfrac: volume fraction or density of spheres in the volume (usually between 0.1 and 0.55).
* r: radius of particles in pixels in the image.
* particle_size: In micrometers. we use this to define how small the particles would look through the microscope (between 0.1 and 1 micrometers), this determines how bad the point spread function is in the simulation.
* brightness: particle brightness usually between 30-255 (8bit grayscale values). Also known as f_mean.
* SNR: signal to noise ratio.
* CNR: contrast to noise ratio.
* b_sigma and f_sigma, standard deviations of foreground and background noise (please read on the contrast to noise ratio equation above).

The y data is a csv file full of particle positions, please use the read_y function which shows you how to process the csv in the website's format.

To read the data use the following

```python
from read import read_x, read_y
import pandas as pd
import matplotlib.pyplot as plt

#Define paths and index
x_path = "path/to/x_train.hdf5"
y_path = "path/to/y_train.csv"
metadata_path = "path/to/x_train_metadata.csv"
index = 0

# read x, y, and metadata at index
x = read_x(x_path, index)
y = read_y(y_path, index)
metadata_df = pd.read_csv(metadata_path, index_col=0)
metadata = metadata_df.iloc[index].to_dict()
diameter = metadata['r']*2
```