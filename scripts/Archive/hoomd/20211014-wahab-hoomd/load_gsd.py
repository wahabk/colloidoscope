import matplotlib.pyplot as plt
import gsd.hoomd


fn = 'hs-phi-560.gsd'
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
