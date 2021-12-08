import os
import sys
from typing import Any

import hoomd
import hoomd.hpmc
import gsd.hoomd
import numpy as np

def hooomd_sim_positions(phi:int, canvas_size:tuple, diameter:int=10) -> np.ndarray:
	hoomd.context.initialize("--mode=cpu");

	phi_target = phi
	# width of cell for lattice with 4 particles in each cube
	# with 4 particles in each cell
	cell_shape=(diameter*2,diameter*2,diameter*2)
	nx, ny, nz = cell_shape  # n = nx * ny * nz * 4
	# a determines gap between cells
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

	hoomd.run(2)

	#round up to avoid floating point error in while loop
	while round(phi_current, 2) < phi_target:
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

	# if save:
	# d = hoomd.dump.gsd(f"hs-phi-{phi_target * 1000:.0f}.gsd", period=2, group=hoomd.group.all(), overwrite=True)

	#make 100 frames then sample
	hoomd.run(100)

	# take snapshot to extract data at the end
	shape = (system.box.Lz, system.box.Ly, system.box.Lx,)
	snap = system.take_snapshot()
	n_particles = snap.particles.N
	positions = snap.particles.position #from -16 to + 16

	return positions

def convert_hoomd_positions(positions, diameter):
	# convert positions from physics to normal
	positions = positions - positions.min()
	positions = positions * diameter
	positions = np.array([p for p in positions if p[0]<canvas_size[0] and p[1]<canvas_size[1] and p[2]<canvas_size[2]])
	return positions

if __name__ == '__main__':
	positions = hooomd_sim_positions(phi=0.4, canvas_size=(32,128,128), diameter=10)
	positions = convert_hoomd_positions(positions, diameter=10)
	print(type(positions))
	print(positions.shape, positions.max(), positions.min())

