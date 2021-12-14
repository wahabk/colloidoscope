from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions
from colloidoscope import DeepColloid
from colloidoscope.simulator import simulate
import numpy as np
from magicgui import magicgui
from napari.layers import Image
import napari

# init magicgui parameters for sliders
@magicgui(
	call_button='Simulate', 
	r={"widget_type": "Slider", 'maximum': 30},
	xy_gauss={"widget_type": "Slider", 'maximum': 10},
	z_gauss={"widget_type": "Slider", 'maximum': 10},
	brightness={"widget_type": "Slider", 'maximum': 255},
	noise={"widget_type": "FloatSlider", 'maximum': 0.1},
	label={"widget_type": "CheckBox"},
	layout='vertical',)
def update_simulation(layer:Image, r:int=5, 
						xy_gauss:int=1, z_gauss:int=2, brightness:int=255, 
						noise:float=0, label:bool=False) -> Image:
	if layer is not None:
		assert isinstance(layer.data, np.ndarray)  # it will be!

		array = layer.data
		canvas_shape = array.shape
		hoomd_centers = layer.metadata['centers']
		centers = convert_hoomd_positions(positions = hoomd_centers, canvas_size=canvas_shape, diameter=r*2)
		new_array, label_array = simulate(canvas_shape, centers, r, xy_gauss, z_gauss, brightness, noise, make_label=True, num_workers=10)
		
		print(new_array.shape, new_array.max(), new_array.min(), r, centers.shape)
		print(label_array.shape, label_array.max(), label_array.min(), r, centers.shape)
		
		if label: layer.data = label_array*255
		else: layer.data = new_array
		
		return 


if __name__ == "__main__":
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size = (32,128,128)
	volfrac = 0.1
	
	canvas = np.zeros(canvas_size, dtype='uint8')
	centers = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size)
	canvas_metadata = {'centers' : centers}

	viewer = napari.Viewer()
	viewer.add_image(canvas, name="Simulated colloids", metadata=canvas_metadata)
	# Add it to the napari viewer
	viewer.window.add_dock_widget(update_simulation)
	# update the layer dropdown menu when the layer list changes
	viewer.layers.events.changed.connect(update_simulation.reset_choices)
	# napari points tutorial: https://napari.org/tutorials/fundamentals/points.html
	napari.run()


