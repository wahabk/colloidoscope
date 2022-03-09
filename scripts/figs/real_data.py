from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap='gray')
	plt.clf()


if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	path = '/home/wahab/Data/HDD/Colloids/Real/Levke/Levke_smallParticles_betterData_2021_4_1/goodData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	array = dc.read_tif(path)
	print(path, array.shape)
	array = array[0]
	array = dc.crop3d(array, (128,128,128))
	plot_with_side_view(array, 'output/figs/real_data/Levke.png')


	path = '/home/wahab/Data/HDD/Colloids/Real/Abraham/20210201_emulsion_centres.tif'
	array = dc.read_tif(path)
	print(path, array.shape)
	array = dc.crop3d(array, (128,128,128))
	plot_with_side_view(array, 'output/figs/real_data/Abraham.png')

	path = '/home/wahab/Data/HDD/Colloids/Real/Emily/0_45-newNewGood.lif - Series021.tif'
	array = dc.read_tif(path)
	print(path, array.shape)
	array = dc.crop3d(array, (128,128,128))
	plot_with_side_view(array, 'output/figs/real_data/Emily.png')

	path = '/home/wahab/Data/HDD/Colloids/Real/Katherine/5khz_5V_d1.tif'
	array = dc.read_tif(path)
	array = array[0]
	print(path, array.shape)
	array = dc.crop3d(array, (64,64,64))
	plot_with_side_view(array, 'output/figs/real_data/Katherine.png')

