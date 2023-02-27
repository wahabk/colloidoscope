from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
import psf
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import convolve



# @njit()
def draw_mask_slice(args):
	# extract args
	s, z, r, centers = args
	#initiate new slice to be drawn
	new_slice = s
	#for each sphere check if this pixel is inside it
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				# euclidean distance
				dist = math.sqrt((z - cz)**2 + (i - cy)**2 + (j - cx)**2)
				
				if dist <= r:
					new_slice[i,j] = 1
	return new_slice

def make_mask(array, centers, diameter, num_workers=1):
	r = int(np.floor(np.min(diameter)/2))
	canvas = np.zeros(array.shape)
	print(r, canvas.shape)
	args = [(s, z, r, centers) for z, s in enumerate(canvas)]

	mask = []
	with tqdm(total=len(args)) as pbar:
		with ProcessPoolExecutor(max_workers=num_workers) as pool:
			for i in pool.map(draw_mask_slice, args):
				mask.append(i)
				pbar.update(1)

	mask = list(mask)
	mask = np.array(mask, dtype='uint8')
	return mask

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside)
	return sidebyside

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	r = 10 #pizels
	particle_size = 0.2
	image = np.zeros((64,64,64), dtype="uint8")

	projections = []

	image = make_mask(image, [[32,32,32],], r*2)*255
	proj = plot_with_side_view(image, "output/figs/psfs/image.png")
	projections.append(proj/proj.max())


	gauss = ndimage.gaussian_filter(image, (9,5,5))
	proj = plot_with_side_view(gauss, "output/figs/psfs/gauss.png")
	projections.append(proj/proj.max())

	# gbl
	args = dict(shape=(64, 64), dims=(particle_size, particle_size), 
				ex_wavelen=488, em_wavelen=520, num_aperture=1.2, refr_index=1.4, 
				pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	this_kernel = obsvol.volume()
	this_kernel = ndimage.zoom(this_kernel, 0.3)
	this_kernel = this_kernel/this_kernel.max()
	gbl = convolve(image, this_kernel, mode='same')
	# dc.view(gbl)
	proj = plot_with_side_view(gbl, "output/figs/psfs/gbl.png")
	projections.append(proj/proj.max())


	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()
	nm_pixel = (particle_size*1000) / r 
	psf_nm_pixel = 30 # standard_huygens_pixel_nm
	psf_zoom = psf_nm_pixel / nm_pixel
	this_kernel = ndimage.zoom(psf_kernel, psf_zoom)
	huygens = convolve(image, this_kernel, mode='same')
	# dc.view(huygens)
	proj = plot_with_side_view(huygens, "output/figs/psfs/huygensXY.png")
	projections.append(proj/proj.max())

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()
	nm_pixel = (particle_size*1000) / r 
	psf_nm_pixel = 30 # standard_huygens_pixel_nm
	psf_zoom = psf_nm_pixel / nm_pixel
	this_kernel = ndimage.zoom(psf_kernel, psf_zoom)
	huygens = convolve(image, this_kernel, mode='same')
	# dc.view(huygens)
	proj = plot_with_side_view(huygens, "output/figs/psfs/huygens50.png")
	projections.append(proj/proj.max())

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedZ.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	# dc.view(psf_kernel)
	print(psf_kernel.shape)
	psf_kernel = dc.crop3d(psf_kernel, (48,16,16), (24,159,160))
	# dc.view(psf_kernel)

	psf_kernel = psf_kernel/psf_kernel.max()
	nm_pixel = (particle_size*1000) / r 
	psf_nm_pixel = 30 # standard_huygens_pixel_nm
	psf_zoom = psf_nm_pixel / nm_pixel
	this_kernel = ndimage.zoom(psf_kernel, psf_zoom)
	huygens = convolve(image, this_kernel, mode='same')
	# dc.view(huygens)
	proj = plot_with_side_view(huygens, "output/figs/psfs/huygensZ.png")
	projections.append(proj/proj.max())

	projections = np.array(projections)
	big_img = np.concatenate(projections, axis=1)

	plt.imsave("output/figs/psfs/all.png", big_img)
