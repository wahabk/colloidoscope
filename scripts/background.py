import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import napari
import numpy as np
from random import randint
from perlin_numpy import generate_perlin_noise_3d
from scipy import ndimage
import math


def make_background_old(canvas_size, octaves, brightness, dtype='uint8'):
	
	from perlin_noise import PerlinNoise

	noise = PerlinNoise(octaves=octaves)
	zpix, xpix, ypix = canvas_size
	pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)] 
	pic = [pic for _ in range(zpix)]

	pic = np.array(pic)
	pic = pic-pic.min()
	pic = pic/pic.max()
	pic = pic*brightness

	pic = np.array(pic, dtype=dtype)
	return pic

def make_background(canvas_shape, octaves, brightness_mean, brightness_std, tileable=(False, False, False)):
    array = generate_perlin_noise_3d(canvas_shape, octaves)

    # normalise to 0 - 1
    array = array-array.min()
    array = array/array.max()

    # find linear transformation to shift mean and std to required
    a = math.sqrt( brightness_std ** 2 / array.std() ** 2 )
    b = brightness_mean - (a * array.mean())
    array = a * array + b

    array = ndimage.gaussian_filter(array, (3,3,3))
    
    return array

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap='gray')
	plt.clf()


canvas_size = (64, 64, 64)
array = make_background(canvas_size, (4,4,4), 200, 20)

print(array.max(), array.min(), array.mean(), array.std())

# napari.view_image(array)
# napari.run()
plt.hist(array.flatten(), )
plt.show()


plot_with_side_view(array, 'output/test_background.png')