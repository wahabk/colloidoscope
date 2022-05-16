from colloidoscope import hoomd_make_configurations, read_gsd, hoomd_make_configurations_polydisp
from colloidoscope import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, uniform
import hoomd.hpmc
import gsd.hoomd

if __name__ == '__main__':

	# phi = 0.1
	# hoomd_make_configurations_polydisp(phi, n_frames=5, output_folder='/home/ak18001/Data/HDD/Colloids/Positions/test/')
	# path = f'/home/ak18001/Data/HDD/Colloids/Positions/test/phi_{phi*1000:.0f}_poly.gsd'
	# positions, diameters = read_gsd(path, 0)
	# print(positions)

	for phi in np.linspace(0.1,0.55,10):
		phi = round(phi, 2)
		# if phi==0.1: continue
		print(phi)
		hoomd_make_configurations_polydisp(phi, n_frames=250+1, output_folder='/home/ak18001/Data/HDD/Colloids/Positions/poly/')
		# positions, diameters= read_gsd(f'/home/ak18001/Data/HDD/Colloids/Positions/test/phi_{phi*1000:.0f}_poly.gsd', 0)
		# print(positions, diameters)
