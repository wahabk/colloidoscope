from deprecated import deprecated
from colloidoscope import hoomd_make_configurations, read_gsd
from colloidoscope import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import hoomd.hpmc
import gsd.hoomd

@deprecated
def read_gsd_old(file_name):
	fn = file_name 
	gsd_iter = gsd.hoomd.open(fn, 'rb')

	# get particles positions in different frames
	for i, frame in enumerate(gsd_iter):
		print(f"Retrieving data in the {i+1}th frame")

		box = frame.configuration.box[:3]
		positions = frame.particles.position
		diameters = frame.particles.diameter

		print("box is a numpy array with shape: ", box.shape)
		print("positions is a numpy array with shape", positions.shape)
		print("diameters is a numpy array with shape", diameters.shape, "\n")

	gsd_iter.close()

if __name__ == '__main__':

	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'

	# phi = 0.5

	# path = f'output/Positions/phi{phi*1000:.0f}.gsd'
	# positions = read_gsd(path, 0)
	# print(positions)

	# hoomd_make_configurations(phi, n_frames=1+1, output_folder='/home/ak18001/Data/HDD/Colloids/Positions/test/')
	
	n_frames = 500
	phis = np.linspace(0.25,0.55,7)
	output_folder = dataset_path+'Positions/big/'

	for phi in phis:
		phi = round(phi, 2)
		# if phi == 0.1: continue
		print(phi)
		hoomd_make_configurations(phi, n_frames=n_frames+1, output_folder=output_folder)
		# positions = read_gsd(f'{output_folder}/phi{phi*1000:.0f}.gsd')
