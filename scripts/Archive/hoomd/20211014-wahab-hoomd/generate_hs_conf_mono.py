import os
import sys

import hoomd
import hoomd.hpmc
import gsd.hoomd
import numpy as np


hoomd.context.initialize("--mode=cpu");

phi_target = 0.5

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

d = hoomd.dump.gsd(f"hs-phi-{phi_target * 1000:.0f}.gsd", period=100, group=hoomd.group.all(), overwrite=True)

hoomd.run(1000)
