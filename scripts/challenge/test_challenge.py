import colloidoscope
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids/challenge/'
	dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = colloidoscope.DeepColloid(dataset_path)

	from read import read_x, read_y
	from metric import average_precision, plot_pr, remove_borders
	
	path = ""
	x_path = dataset_path+"challenge/x_test.hdf5"
	m_path = dataset_path+"challenge/x_test_metadata.csv"
	y_path = dataset_path+"challenge/y_test.csv"
	index=0
	
	x = read_x(x_path, index)
	y = read_y(y_path, index)
	metadata = pd.read_csv(m_path, index_col=0)
	print(y.shape)
	# exit()

	metadata = metadata.iloc[index].to_dict()
	diameter = metadata['r']*2
	
	df, pred, label = dc.detect(x, diameter=diameter, 
                                weights_path='output/weights/attention_unet_202206.pt', 
                                patch_overlap=(16,16,16),
                                debug=True, run_on="cpu",
                                post_processing="log", batch_size=1)
	

	
	ap, precisions, recalls = average_precision(y, pred, diameter=diameter, canvas_size=x.shape)
	fig = plot_pr(ap, precisions, recalls, name='Unet', tag='o-', color='red')
	# ap, precisions, recalls, thresholds = average_precision(y, tp_pred, diameter=diameter)
	# fig = plot_pr(ap, precisions, recalls, thresholds, name='trackpy', tag='x-', color='gray')
	plt.show()
	
	dc.view(array=x, positions=pred, label=label)
	
