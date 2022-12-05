import colloidoscope
from read import read_x, read_y
import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np

if __name__ == '__main__':
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids/challenge/'
	# dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = colloidoscope.DeepColloid(dataset_path)

	from read import read_x, read_y
	from metric import average_precision, plot_pr, exclude_borders
	
	path = ""
	x_path = dataset_path+"challenge/x_test.hdf5"
	m_path = dataset_path+"challenge/x_test_metadata.csv"
	y_path = dataset_path+"challenge/y_test.csv"
	y_path_new = dataset_path+"challenge/y_test_new.csv"

	df = pd.read_csv(y_path)

	df = df.rename(columns={"index": "image_index"})

	print(df)

	df.to_csv(y_path_new)


