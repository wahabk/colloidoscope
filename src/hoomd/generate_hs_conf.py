import os
import sys
import itertools

import hoomd
import hoomd.hpmc
import gsd.hoomd
import numpy as np



"""
Generating Lattice
"""
phi_target = 0.1  # the volume fraction of the system after compression
n_particle = 1000
poly_dispersity = 0.1
diameter = 1.0
n_type = 10


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

init_fn = 'lattice-poly.gsd'
if init_fn in os.listdir('.'):
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

while phi_current < phi_target:
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

d = hoomd.dump.gsd(f"hs-phi-{phi_target * 1000:.0f}.gsd", period=100, group=hoomd.group.all(), overwrite=True)

"""
Sample Configurations
"""
hoomd.run(10_000)
