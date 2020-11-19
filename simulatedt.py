import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer
from concurrent.futures import ProcessPoolExecutor
import random
from math import sqrt
import deeptrack as dt

IMAGE_SIZE = 64

particle = dt.PointParticle(
    position=lambda: np.random.rand(2) * IMAGE_SIZE,
    position_unit="pixel",
    intensity=1
)

fluorescence_microscope = dt.Fluorescence(
    NA=0.8,
    wavelength=680e-9,
    magnification=10,
    resolution=1e-6,
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE)
)

imaged_particle = fluorescence_microscope(particle)

imaged_particle.update()
output_image = imaged_particle.resolve()

plt.imshow(np.squeeze(output_image), cmap='gray')
plt.show()

