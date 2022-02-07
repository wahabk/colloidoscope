from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions
from colloidoscope import DeepColloid
from colloidoscope.simulator import simulate
from colloidoscope.hoomd_sim_positions import read_gsd
import numpy as np
from magicgui import magicgui
from napari.layers import Image
import napari

# init magicgui parameters for sliders
@magicgui(
	call_button='Simulate', 
	r={"widget_type": "Slider", 'max': 30},
	xy_gauss={"widget_type": "Slider", 'max': 30},
	z_gauss={"widget_type": "Slider", 'max': 30},
	max_brightness={"widget_type": "Slider", 'max': 255, 'min':150},
	min_brightness={"widget_type": "Slider", 'max': 150, 'min':50},
	noise={"widget_type": "FloatSlider", 'max': 0.2},
	layout='vertical',)
def update_simulation(layer:Image, label_layer:Image, r:int=6, 
						xy_gauss:int=2, z_gauss:int=5, max_brightness:int=255, 
						min_brightness:int=75, noise:float=0.01) -> Image:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!

		array = layer.data
		canvas_size = array.shape
		hoomd_positions = layer.metadata['hoomd_positions']
		hoomd_diameters = layer.metadata['hoomd_diameters']

		centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=hoomd_diameters)
		
		new_array, label_array = simulate(canvas_size, centers, r, xy_gauss, z_gauss, min_brightness, max_brightness, 
							noise, make_label=True, diameters=diameters, num_workers=10)

		print(new_array.shape, new_array.max(), new_array.min(), r, centers.shape)
		print(label_array.shape, label_array.max(), label_array.min(), r, centers.shape)
		
		layer.data = new_array
		if label_layer: label_layer.data = label_array*255
		
		return 


if __name__ == "__main__":
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size = (32,128,128)
	volfrac = 0.3
	centers_path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
	
	canvas = np.zeros(canvas_size, dtype='uint8')
	hoomd_positions, hoomd_diameters = read_gsd(centers_path, 1)
	canvas_metadata = {'hoomd_diameters' : hoomd_diameters,
						'hoomd_positions' : hoomd_positions}

	viewer = napari.Viewer()
	viewer.add_image(canvas, name="Simulated colloids", metadata=canvas_metadata)
	viewer.add_image(canvas, name="Simulated labels", metadata=canvas_metadata, opacity=0.5, colormap='red')
	# Add it to the napari viewer
	viewer.window.add_dock_widget(update_simulation)
	# update the layer dropdown menu when the layer list changes
	viewer.layers.events.changed.connect(update_simulation.reset_choices)
	# napari points tutorial: https://napari.org/tutorials/fundamentals/points.html
	napari.run()


