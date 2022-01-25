from colloidoscope import hoomd_make_configurations, read_gsd, hoomd_make_configurations_polydisp
from colloidoscope import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import hoomd.hpmc
import gsd.hoomd

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

	# phi = 0.1
	# hoomd_make_configurations_polydisp(phi, n_frames=5, output_folder='/home/ak18001/Data/HDD/Colloids/Positions/test/')
	# path = f'/home/ak18001/Data/HDD/Colloids/Positions/test/phi_{phi*1000:.0f}_poly.gsd'
	# positions, diameters = read_gsd(path, 0)
	# print(positions)

	for phi in np.linspace(0.1,0.5,5):
		phi = round(phi, 2)
		if phi==0.1: continue
		print(phi)
		hoomd_make_configurations_polydisp(phi, n_frames=100, output_folder='/home/ak18001/Data/HDD/Colloids/Positions/test/')
		# positions, diameters= read_gsd(f'/home/ak18001/Data/HDD/Colloids/Positions/test/phi_{phi*1000:.0f}_poly.gsd', 0)
		# print(positions, diameters)
