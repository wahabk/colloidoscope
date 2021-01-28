import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer
import trackpy as tp
import explore_lif
from make_dataset import *

# reader = explore_lif.Reader('Data/i-ii.lif')
# series = reader.getSeries()
# chosen = series[5]  # choose first image in the lif file
# # get a numpy array corresponding to the 1st time point & 1st channel
# # the shape is (x, y, z)
# video = [chosen.getXYZ(T=t, channel=0) for t in range(chosen.getNbFrames())]
# video = np.array(video)
# print(video.shape)
# t, x, y, z

canvas, positions = read_hdf5('First', 1, positions=True)

f = tp.locate(canvas, 11)
f = [list(f[:]['z']), list(f[:]['y']), list(f[:]['x'])]

new = []
for i in range(0, len(f[0])):
	new.append([ f[0][i], f[1][i], f[2][i] ])
f = np.array(new)

f = np.sort(f)
positions = np.sort(positions)


print(f.astype(int))
print(positions.astype(int))

make_gif()

