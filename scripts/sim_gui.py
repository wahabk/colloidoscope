from numba import njit
import numpy as np
import random
import os
from skimage.util import random_noise
import time
import random
import os
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions
from colloidoscope import DeepColloid
from colloidoscope.simulator import sim_from_parameters
from magicgui import magicgui
import napari
from napari.types import ImageData

# init magicgui parameters for sliders
@magicgui(
	call_button='update', 
	r={"widget_type": "Slider", 'maximum': 30},
	zoom={"widget_type": "FloatSlider", 'maximum': 1},
	xy_gauss={"widget_type": "Slider", 'maximum': 20},
	z_gauss={"widget_type": "Slider", 'maximum': 20},
	brightness={"widget_type": "Slider", 'maximum': 255},
	noise={"widget_type": "FloatSlider", 'maximum': 1},
	layout='horizontal',)
def update_simulation(layer: ImageData, r:int, zoom:float, 
						xy_gauss:int, z_gauss:int, brightness:int, 
						noise=float) -> ImageData:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!

		array = layer.data
		canvas_shape = array.shape
		hoomd_centers = layer.metadata['centers']
		centers = convert_hoomd_positions(hoomd_centers, diameter=r*2)
		new_array = sim_from_parameters(canvas_shape, centers, r, zoom, xy_gauss, z_gauss, brightness, noise)

		return new_array


if __name__ == "__main__":
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size = (32,128,128)
	volfrac = 0.2
	
	
	canvas = np.zeros(canvas_size, dtype='uint8')
	centers = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size, diameter=10)
	canvas_metadata = {'centers' : centers}

	viewer = napari.Viewer()
	viewer.add_image(canvas, name="Simulated colloids", metadata=canvas_metadata)

	# Add it to the napari viewer
	viewer.window.add_dock_widget(update_simulation)
	# update the layer dropdown menu when the layer list changes
	viewer.layers.events.changed.connect(update_simulation.reset_choices)

	# napari points tutorial: https://napari.org/tutorials/fundamentals/points.html

	napari.run()


