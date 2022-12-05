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
	y_path = dataset_path+"challenge/y_test_new.csv"

	big_d = pd.DataFrame(columns=['image_index', "X", "Y", "Z"])

	for index in range(10):
		print(index)
		
		x = read_x(x_path, index)
		y = read_y(y_path, index)
		metadata = pd.read_csv(m_path, index_col=0)
		metadata = metadata.iloc[index].to_dict()
		diameter = metadata['r']*2
		
		tp_pred, df = dc.run_trackpy(x, diameter=dc.round_up_to_odd(diameter))
		print(len(y),  len(tp_pred))
		len_nans = len(y) - len(tp_pred)
		nans = np.empty((len_nans, 3))
		nans[:] = np.nan
		tp_pred = np.concatenate([tp_pred, nans], axis=0)
		print(len(y),  len(tp_pred))

		indices = np.full((len(tp_pred),1), fill_value=int(index))
		positions_with_indices = np.concatenate([indices, tp_pred], axis=1)
		pos_df = pd.DataFrame(positions_with_indices)
		pos_df.columns = ["image_index", "Z", "X", "Y"]

		big_d = pd.concat([big_d, pos_df], axis=0, ignore_index=True)
	
	big_d.to_csv("output/y_benchmark.csv")
