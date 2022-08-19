import numpy as np
import cv2
from skimage.util.shape import view_as_windows
from skimage import io
from pathlib2 import Path
from colloidoscope.deepcolloid import DeepColloid


if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	weights_path = "output/weights/attention_unet_202206.pt"
	examples_path = "examples/Data/levke.tiff"
	array = dc.read_tif(examples_path)
	print(array.shape)

	df, positions, label = dc.detect(array, diameter=11, weights_path=weights_path)
 
	print(df)

	



