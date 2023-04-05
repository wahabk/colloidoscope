from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
from scipy import ndimage
from skimage import io

def plot_with_side_view(scan):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside

def read_real_examples():

	d = {}

	d["A - Silica (560nm 0.55Φ)"] = {}
	d["A - Silica (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["A - Silica (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["A - Silica (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/james.tiff')
	d["B - Silica Decon (560nm 0.55Φ)"] = {}
	d["B - Silica Decon (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["B - Silica Decon (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["B - Silica Decon (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/jamesdecon.tiff')
	# d["E - Silica (500nm 0.50Φ) "] = {}
	# d["E - Silica (500nm 0.50Φ) "]['diameter'] = 13
	# d["E - Silica (500nm 0.50Φ) "]['volfrac'] = 0.5
	# d["E - Silica (500nm 0.50Φ) "]['array'] = io.imread('examples/Data/emily.tiff')
	d["C - PMMA (315nm 0.58Φ)"] = {}
	d["C - PMMA (315nm 0.58Φ)"]['diameter'] = [15,11,11]
	d["C - PMMA (315nm 0.58Φ)"]['volfrac'] = 0.58
	d["C - PMMA (315nm 0.58Φ)"]['array'] = io.imread('examples/Data/levke.tiff')
	d["D - Emulsion (3μm 0.64Φ)"] = {}
	d["D - Emulsion (3μm 0.64Φ)"]['diameter'] = 15
	d["D - Emulsion (3μm 0.64Φ)"]['volfrac'] = 0.64
	array = io.imread('examples/Data/abraham.tiff') 
	d["D - Emulsion (3μm 0.64Φ)"]['array'] = ndimage.zoom(array, 2.25)
	d["E - Silica (1.2μm 0.2Φ)"] = {}
	d["E - Silica (1.2μm 0.2Φ)"]['diameter'] = 15
	d["E - Silica (1.2μm 0.2Φ)"]['volfrac'] = 0.2
	array  = io.imread('examples/Data/katherine.tiff')
	d["E - Silica (1.2μm 0.2Φ)"]['array'] = ndimage.zoom(array, 2)

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


	# conIndex = 1
	# deconIndex = 2
	# path = dataset_path + '/Real/James/NBD_CS2_vf58_day6_time_2.lif'
	# reader = dc.Explore_lif_reader(path)
	# index = reader.chooseSerieIndex()
	# serie = reader.getSeries()[conIndex]
	# array = serie.getXYZ().T
	# print(path)
	# print(array.shape)
	# dc.view(array)
	# imsave('examples/Data/james.tiff', array)
	# James = plot_with_side_view(array, 'output/figs/real_data/james.png')

	# path = dataset_path + '/Real/James/NBD_CS2_vf58_day6_time_2_decon.lif'
	# reader = dc.Explore_lif_reader(path)
	# index = reader.chooseSerieIndex()
	# serie = reader.getSeries()[index]
	# array = serie.getXYZ().T
	# print(path)
	# print(array.shape)
	# dc.view(array)
	# imsave('examples/Data/jamesdecon.tiff', array)
	# Jamesdecon = plot_with_side_view(array, 'output/figs/real_data/jamesdecon.png')
	
	real_dict = read_real_examples()
 
	fig, axs = plt.subplots(2,len(real_dict),sharex=True,sharey=True)
	plt.tight_layout()

	for i, (name, d) in enumerate(real_dict.items()):
		array = d['array']
		array = dc.crop3d(array, (100,100,100))
		array = ndimage.zoom(array,2)
		print(name, array.shape, i)
  
		projection = np.max(array, axis=0)
		axs[0,i].imshow(projection)#, cmap='gray')
		axs[0,i].set_title(name,fontsize=11)
		axs[0,i].set_xticks([])
		axs[0,i].set_yticks([])
		if i == 0: axs[0,i].set_xlabel("X")
		if i == 0: axs[0,i].set_ylabel("Y")
		projection = np.max(array, axis=1)
		axs[1,i].imshow(projection)#, cmap='gray')
		axs[1,i].set_title("")
		axs[1,i].set_xticks([])
		axs[1,i].set_yticks([])
		if i == 0: axs[1,i].set_xlabel("X")
		if i == 0: axs[1,i].set_ylabel("Z")
	fig.set_figwidth(12)
	fig.set_figheight(4)
	plt.savefig("output/figs/real_data/real.png", bbox_inches="tight")
  
