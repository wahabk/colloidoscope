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

	real_path = Path(dataset_path) / 'Real'

	for person in real_path.iterdir():
		print(person)

		# file_path = 

	



