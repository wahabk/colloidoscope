from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
from scipy import ndimage
from skimage import io

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside

def read_real_examples():

	d = {}

	d['abraham'] = {}
	d['abraham']['diameter'] = 9
	d['abraham']['array'] = io.imread('examples/Data/abraham.tiff')
	d['emily'] = {}
	d['emily']['diameter'] = 9
	d['emily']['array'] = io.imread('examples/Data/emily.tiff')
	d['katherine'] = {}
	d['katherine']['diameter'] = 9
	d['katherine']['array'] = io.imread('examples/Data/katherine.tiff')
	d['levke'] = {}
	d['levke']['diameter'] = 9
	d['levke']['array'] = io.imread('examples/Data/levke.tiff')

	return d



if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	# path = dataset_path + '/Real/Levke/goodData_2021_4_1Levke_smallParticles_betterData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'
	# array = dc.read_tif(path) 
	# print(path, array.shape)
	# array = array[0]
	# array = ndimage.zoom(array, 1)
	# dc.view(array)
	# print(array.shape)
	# imsave('examples/Data/levke.tiff', array)
	# Levke = plot_with_side_view(array, 'output/figs/real_data/Levke.png')


	# path = dataset_path + '/Real/Abraham/20210201_emulsion_centres.tif'
	# array = dc.read_tif(path)
	# print(path, array.shape)
	# array = ndimage.zoom(array, 0.25)
	# dc.view(array)
	# print(array.shape)
	# imsave('examples/Data/abraham.tiff', array)
	# Abraham = plot_with_side_view(array, 'output/figs/real_data/Abraham.png')


	# path = dataset_path + '/Real/Emily/0_45-newNewGood.lif - Series021.tif'
	# array = dc.read_tif(path)
	# print(path, array.shape)
	# dc.view(array)
	# # array = dc.crop3d(array, (128,128,128))
	# print(array.shape)
	# imsave('examples/Data/emily.tiff', array)
	# Emily = plot_with_side_view(array, 'output/figs/real_data/Emily.png')


	# path = dataset_path + '/Real/Katherine/5khz_5V_d1.tif'
	# array = dc.read_tif(path)
	# array = array[0]
	# print(path, array.shape)
	# dc.view(array)
	# # array = dc.crop3d(array, (64,64,64))
	# # array = ndimage.zoom(array, 2)
	# print(array.shape)
	# imsave('examples/Data/katherine.tiff', array)
	# Katherine = plot_with_side_view(array, 'output/figs/real_data/Katherine.png')


	conIndex = 1
	deconIndex = 2
	path = dataset_path + '/Real/James/NBD_CS2_vf58_day6_time_2.lif'
	reader = dc.Explore_lif_reader(path)
	index = reader.chooseSerieIndex()
	serie = reader.getSeries()[conIndex]
	array = serie.getXYZ().T
	print(path)
	print(array.shape)
	dc.view(array)
	imsave('examples/Data/james.tiff', array)
	James = plot_with_side_view(array, 'output/figs/real_data/james.png')

	path = dataset_path + '/Real/James/NBD_CS2_vf58_day6_time_2_decon.lif'
	reader = dc.Explore_lif_reader(path)
	index = reader.chooseSerieIndex()
	serie = reader.getSeries()[index]
	array = serie.getXYZ().T
	print(path)
	print(array.shape)
	dc.view(array)
	imsave('examples/Data/jamesdecon.tiff', array)
	Jamesdecon = plot_with_side_view(array, 'output/figs/real_data/jamesdecon.png')
	