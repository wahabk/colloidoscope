import colloidoscope
from read import read_x, read_y
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = colloidoscope.DeepColloid(dataset_path)

	from read import read_x, read_y
	from metric import average_precision, plot_pr, exclude_borders
	
	path = ""
	x_path = "x_train.hdf5"
	m_path = "x_train_metadata.csv"
	y_path = "y_train.csv"
	index=0
	
	x = read_x(x_path, index)
	y = read_y(y_path, index)
	metadata = pd.read_csv(m_path, index_col=0)
	metadata = metadata.iloc[index].to_dict()
	diameter = metadata['r']*2
	
	df, pred = dc.detect(x, diameter=diameter, 
                                weights_path='output/weights/attention_unet_202206.pt', 
                                patch_overlap=(16,16,16),
                                debug=True, device="cpu",
                                post_processing="log", batch_size=1)
	
	tp_pred = dc.run_trackpy(x, diameter=diameter)
	
	ap, precisions, recalls, thresholds = average_precision(y, pred, diameter=diameter)
	fig = plot_pr(ap, precisions, recalls, thresholds, name='Unet', tag='o-', color='red')
	ap, precisions, recalls, thresholds = average_precision(y, tp_pred, diameter=diameter)
	fig = plot_pr(ap, precisions, recalls, thresholds, name='trackpy', tag='x-', color='gray')
	plt.show()
	
	
