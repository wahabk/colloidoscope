import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer
import trackpy as tp
import explore_lif

reader = explore_lif.Reader('Data/i-ii.lif')
series = reader.getSeries()
chosen = series[5]  # choose first image in the lif file
# get a numpy array corresponding to the 1st time point & 1st channel
# the shape is (x, y, z)
video = [chosen.getXYZ(T=t, channel=0) for t in range(chosen.getNbFrames())]
video = np.array(video)
print(video.shape)
# t, x, y, z

f = tp.locate(video[0], 11)
print(f)

