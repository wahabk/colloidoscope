from colloidoscope import hoomd_sim_positions
from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions
from colloidoscope import DeepColloid
from colloidoscope.simulator import simulate
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np
import random

import hoomd
import hoomd.hpmc
import gsd.hoomd

import sys

def hoomd_make_configurations(phi:float, n_frames=100, output_folder='output/Positions/'):
	
	hoomd.context.initialize("--mode=cpu");

	phi_target = phi

	sample_period = 100
	nx, ny, nz = 20, 20, 20  # n = nx * ny * nz * 4
	a = 3.5

	system = hoomd.init.create_lattice(
		unitcell=hoomd.lattice.fcc(a=a), n=[nx, ny, nz],
	)

	mc = hoomd.hpmc.integrate.sphere(seed=0, d=0.1)
	mc.shape_param.set('A', diameter=1.0);

	# compress
	N = len(system.particles)
	phi_current = N / 6 * np.pi / system.box.get_volume()
	V_target = N / 6 * np.pi / phi_target
	dV = 0.9

	hoomd.run(100)

	while phi_current < phi_target:
		V_current = max(system.box.get_volume() * dV, V_target);
		new_box = system.box.set_volume(V_current);
		hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None);
		phi_current = N / 6 * np.pi / V_current
		overlaps = mc.count_overlaps()
		while overlaps > 0:
			hoomd.run(10, quiet=True);
			overlaps = mc.count_overlaps()
			print(overlaps, end=' ');
			sys.stdout.flush();
		print("compressing", phi_current, " -> ", phi_target)

	d = hoomd.dump.gsd(f"{output_folder}phi{phi_target*1000:.0f}.gsd", period=sample_period, group=hoomd.group.all(), overwrite=True)

	hoomd.run(n_frames*sample_period)

def read_gsd(file_name):
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


	for phi in np.linspace(0.1,0.55,10,dtype='float32'):
		print(phi)

		hoomd_make_configurations(phi, n_frames=1000)
		positions = read_gsd(f'output/Positions/phi{phi*1000:.0f}.gsd')