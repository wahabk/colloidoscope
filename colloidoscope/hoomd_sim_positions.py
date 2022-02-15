import os
import sys

import numpy as np

import itertools

def hooomd_sim_positions(phi:int, canvas_size:tuple) -> np.ndarray:
	import hoomd
	import hoomd.hpmc

	hoomd.context.initialize("--mode=cpu");

	phi_target = phi
	# width of cell for lattice with 4 particles in each cube
	# with 4 particles in each cell
	cell_shape=(10*2,10*2,10*2)
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

	hoomd.run(2, quiet=True)

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
			# print(overlaps, end=' ');
			sys.stdout.flush();
		# print("compressing", phi_current, " -> ", phi_target)

	# if save:
	# d = hoomd.dump.gsd(f"hs-phi-{phi_target * 1000:.0f}.gsd", period=2, group=hoomd.group.all(), overwrite=True)

	#make 100 frames then sample
	hoomd.run(50, quiet=True)

	# take snapshot to extract data at the end
	shape = (system.box.Lz, system.box.Ly, system.box.Lx,)
	snap = system.take_snapshot()
	n_particles = snap.particles.N
	positions = snap.particles.position #from -16 to + 16

	return positions

def hoomd_make_configurations(phi:float, n_frames=100, output_folder='output/Positions/'):
	
	import hoomd
	import hoomd.hpmc

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

	d = hoomd.dump.gsd(f"{output_folder}phi{phi_target*1000:.0f}.gsd", period=sample_period, group=hoomd.group.all(), overwrite=True)

	hoomd.run(n_frames*sample_period)

def hoomd_make_configurations_polydisp(phi:float, n_frames=100, output_folder='output/Positions/'):

	import hoomd
	import hoomd.hpmc
	import gsd.hoomd

	"""
	Generating Lattice
	"""
	phi_target = phi  # the volume fraction of the system after compression
	n_particle = 6000
	poly_dispersity = 0.1
	diameter = 1.0
	n_type = 10
	sample_period = 100


	"""
	Generating Lattice
	"""
	m = np.ceil(np.power(n_particle / 4, 1.0/3.0))
	n_site_fcc = 4 * m ** 3
	spacing = 2.0

	K = np.ceil(n_particle**(1 / 3)).astype(int)
	L = K * spacing
	x = np.linspace(-L / 2, L / 2, K, endpoint=False)
	position = list(itertools.product(x, repeat=3))


	"""
	create the system
	"""
	snapshot = gsd.hoomd.Snapshot()
	snapshot.particles.N = n_particle
	snapshot.particles.position = position[0 : n_particle]

	# put poly dispersed diameters into separate bins
	diameters = np.random.normal(loc=diameter, scale=poly_dispersity, size=n_particle)
	counts, diameter_bin_edges = np.histogram(diameters, bins=n_type)
	diameter_bin_centre = (diameter_bin_edges[1:] + diameter_bin_edges[:-1]) / 2

	diameters_binned = np.concatenate([
		np.ones(c, dtype=float) * sigma for sigma, c in zip(diameter_bin_centre, counts)
	])

	p_typeid = np.concatenate([
		[i] * c for i, c in zip(range(n_type), counts)
	], axis=0)

	p_types = np.concatenate([
		[f'H{i+1}'] * c for i, c in zip(range(n_type), counts)
	], axis=0)


	snapshot.particles.typeid = p_typeid
	snapshot.particles.types = p_types
	snapshot.particles.diameter = diameters_binned


	snapshot.configuration.box = [L, L, L, 0, 0, 0]

	init_fn = f'{output_folder}lattice_poly.gsd'
	if init_fn in os.listdir(output_folder):
		os.remove(init_fn)
	with gsd.hoomd.open(name=init_fn, mode='xb') as f:
		f.append(snapshot)


	"""
	Starting Simulation
	"""
	hoomd.context.initialize("--mode=cpu --nthreads=64");
	system = hoomd.init.read_gsd(init_fn)

	for i, p in enumerate(system.particles):
		p.type = p_types[i]

	mc = hoomd.hpmc.integrate.sphere(seed=0, d=0.1)
	for i, sigma in enumerate(diameter_bin_centre):
		mc.shape_param.set(f'H{i+1}', diameter=sigma);

	"""
	Compression system to desired volume fraction
	"""
	phi_current = 0
	for sigma, count in zip(diameter_bin_centre, counts):
		print(sigma, count)
		phi_current += count * sigma**3 / 6 * np.pi / system.box.get_volume()


	V_target = 0
	for sigma, count in zip(diameter_bin_centre, counts):
		V_target += count * sigma**3 / 6 * np.pi / phi_target

	dV = 0.9

	hoomd.run(100)

	while round(phi_current, 2) < phi_target:
		V_current = max(system.box.get_volume() * dV, V_target);
		new_box = system.box.set_volume(V_current);
		hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None);
		phi_current = 0
		for sigma, count in zip(diameter_bin_centre, counts):
			phi_current += count * sigma**3 / 6 * np.pi / V_current
		overlaps = mc.count_overlaps()
		while overlaps > 0:
			hoomd.run(100, quiet=True);
			overlaps = mc.count_overlaps()
			print(overlaps, end=' ');
			sys.stdout.flush();
		print("compressing", phi_current, " -> ", phi_target)

	d = hoomd.dump.gsd(f"{output_folder}phi_{phi_target * 1000:.0f}_poly.gsd", period=sample_period, group=hoomd.group.all(), overwrite=True)

	"""
	Sample Configurations
	"""
	hoomd.run(n_frames*sample_period)



def convert_hoomd_positions(positions, canvas_size, diameter=10, diameters=None,):
	# convert positions from physics to normal
	centers = positions.copy()

	# import pdb; pdb.set_trace()
	centers = centers * diameter
	centers = centers - (centers.min(axis=0) + [diameter,diameter,diameter] )
	
	indices = []
	for idx, c in enumerate(centers):
		if diameter/2<c[0]<(canvas_size[0]-diameter/2) and diameter/2<c[1]<(canvas_size[1]-diameter/2) and diameter/2<c[2]<(canvas_size[2]-diameter/2):
			indices.append(idx)

	centers = centers[indices]

	if diameters is not None:
		diameters = diameters[indices]
		print(centers.shape, diameters.shape)
		return centers, diameters
	else:
		return centers

def read_gsd(file_name:str, i:int):
	import gsd.hoomd

	gsd_iter = gsd.hoomd.open(file_name, 'rb')

	num_frames = len(gsd_iter)
	if i >= num_frames:
		raise ValueError

	frame = gsd_iter[i]

	positions = frame.particles.position
	diameters = frame.particles.diameter

	gsd_iter.close()

	return positions, diameters

if __name__ == '__main__':
	positions = hooomd_sim_positions(phi=0.4, canvas_size=(32,128,128), diameter=10)
	positions = convert_hoomd_positions(positions, (32,128,128), diameter=10)
	print(type(positions))
	print(positions.shape, positions.max(), positions.min())

