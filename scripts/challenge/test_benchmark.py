import colloidoscope
from read import read_x, read_y
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	dataset_path = '/home/ak18001/Data/HDD/Colloids/challenge/'
	# dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = colloidoscope.DeepColloid(dataset_path)

	from read import read_x, read_y
	from metric import average_precision, plot_pr, exclude_borders
	
	path = ""
	x_path = dataset_path+"challenge/x_test.hdf5"
	m_path = dataset_path+"challenge/x_test_metadata.csv"
	y_path = dataset_path+"challenge/y_test.csv"

	big_d = pd.DataFrame(columns=['image_index', "X", "Y", "Z"])

	for index in range(10):
		index=0
		
		x = read_x(x_path, index)
		y = read_y(y_path, index)
		metadata = pd.read_csv(m_path, index_col=0)
		metadata = metadata.iloc[index].to_dict()
		diameter = metadata['r']*2
		
		tp_pred, df = dc.run_trackpy(x, diameter=dc.round_up_to_odd(diameter))
		df = df[["X", "Y", "Z"]]

		# tODO concat? add one particle at a time?

		print(df)
	
	df.to_csv("output/y_benchmark.csv")