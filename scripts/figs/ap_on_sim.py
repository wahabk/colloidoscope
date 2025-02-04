from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
import psf
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path

def read_txt(path):
	"""Read txt file containing x,y from the second line in the form

	correlations	55141264
	0	0
	0.025	0.453336
	0.05	0.53576
	0.075	0.816004

	Args:
		path
	"""
	with open(path, 'r') as f:
		lines = f.readlines()
		lines = [l.strip().split() for l in lines]
		lines = np.array(lines[1:], dtype=np.float32)
	return lines

control = read_txt("output/Paper/True_positions.txt")
half_added = read_txt("output/Paper/Half_added.txt")
half_missing = read_txt("output/Paper/Half_missing.txt")
half_missingadded = read_txt("output/Paper/Half_missingadded.txt")

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	dataset_path = '/home/wahab/data/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	volfrac = 0.25
	index = 0
	diameter = 10

	canvas_size = (256,256,256)

	path = f"{dataset_path}/Positions/old/phi{volfrac*1000:.0f}.gsd"
	print(f'Reading: {path} at {index+1} ...')

	hoomd_positions, diameters = read_gsd(path, index)

	true_positions, diameters = convert_hoomd_positions(hoomd_positions, canvas_size=canvas_size, diameters=diameters, diameter=diameter)

	fig, axs = plt.subplots(2,4,  sharey='row')
	plt.tight_layout()

	# P1R1
	true_pos = true_positions
	pred_pos = true_pos

	ap, precisions, recalls, thresholds = dc.average_precision(true_pos, pred_pos, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='True', tag='o-', color='black', axs=axs[0,0], title='A - True positions')
	# ap, precisions, recalls, thresholds = dc.average_precision(true_pos, pred_pos, diameters=diameters)
	# fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Prediction', tag='x-', color='gray', axs=axs[0,0])
	# x, y = dc.get_gr(pred_pos, 100, 100)
	x, y = control.T
	dc.plot_gr(x, y, 1, label=f'True', color='black', axs=axs[1,0])
	axs[1,0].set_ylim(0,2.5)

	# P1R0.5
	true_pos = true_positions
	pred_pos = true_pos[:len(true_pos)//2]

	ap, precisions, recalls, thresholds = dc.average_precision(true_pos, pred_pos, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='Pred', tag='o-', color='red', axs=axs[0,1], title='B - Half missing')
	# x, y = dc.get_gr(pred_pos, 100, 100)
	x, y = half_missing.T
	dc.plot_gr(x, y, 1, label=f'Pred', color='red', axs=axs[1,1])
	axs[1,1].set_ylim(0,2.5)

	plt.savefig("output/figs/metrics_on_sim.png")

	# P0.5R1
	true_pos = true_positions
	pred_pos = true_pos

	anomalies = np.random.random_sample(size=(len(true_pos)//3)*6)
	anomalies = anomalies.reshape((len(anomalies)//3, 3))*canvas_size[0]

	pred_pos = np.concatenate([pred_pos, anomalies], axis=0)

	ap, precisions, recalls, thresholds = dc.average_precision(true_pos, pred_pos, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='Pred', tag='o-', color='green', axs=axs[0,2], title='C - Half added')
	# x, y = dc.get_gr(pred_pos, 100, 100)
	x, y = half_added.T
	dc.plot_gr(x, y, 1, label=f'Pred', color='green', axs=axs[1,2])
	axs[1,2].set_ylim(0,2.5)


	# P0.5R0.5
	true_pos = true_positions
	pred_pos = true_pos[:len(true_pos)//2]

	anomalies = np.random.random_sample(size=(len(true_pos)//3)*3)
	anomalies = anomalies.reshape((len(anomalies)//3, 3))*canvas_size[0]

	pred_pos = np.concatenate([pred_pos, anomalies], axis=0)

	ap, precisions, recalls, thresholds = dc.average_precision(true_pos, pred_pos, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='Pred', tag='o-', color='blue', axs=axs[0,3], title='D - Half missing+added')
	# x, y = dc.get_gr(pred_pos, 100, 100)
	x, y = half_missingadded.T
	dc.plot_gr(x, y, 1, label=f'Pred', color='blue', axs=axs[1,3])
	axs[1,3].set_ylim(0,2.5)

	fig.set_figwidth(12)
	fig.set_figheight(6)
	plt.savefig("output/figs/metrics_on_sim.png")