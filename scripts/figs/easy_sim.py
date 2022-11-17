from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label, exclude_borders
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
from  colloidoscope.predict import find_positions

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	num_workers = 10

	params = dict(
		r=10,
		particle_size=0.45,
		snr=5,
		cnr=1,
		volfrac=0.5,
		brightness=100,
	)

	heatmap_r = 'radius'

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	n = 0
	path = f"{dataset_path}/Positions/old/phi{params['volfrac']*1000:.0f}.gsd"
	print(f'Reading: {path} at {n+1} ...')

	hoomd_positions, diameters = read_gsd(path, n+1)

	heatmap_rs = ["radius", 2, 4, "seg-2", "seg-4"]


	canvas, label, true_positions, diameters = dc.simulate(
				canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
				params['snr'], diameters=diameters, make_label=True, heatmap_r="radius", 
				num_workers=num_workers, psf_kernel=psf_kernel)
	print(true_positions.shape)
	print(canvas.shape, label.shape)


	diameters = diameters * (params['r']*2)
	# does tp/log work best on seg label?
	true_positions, diameters = exclude_borders(true_positions, canvas_size, pad=params['r']/4, diameters=diameters)
	# 

	tp_pred, df = dc.run_trackpy(canvas, diameter = dc.round_up_to_odd(params['r']*2))
	print(dc.round_up_to_odd(params['r']*2))
	print(tp_pred.shape)
	# print(tp_pred, true_positions)
	tp_pred = exclude_borders(tp_pred, canvas_size, pad=params['r']/4)
	print(true_positions.shape)
	print(tp_pred.shape)

	prec, rec = dc.get_precision_recall(true_positions, tp_pred, diameters=diameters, threshold=0.5)
	print('tp', prec, rec)
	# prec, rec = dc.get_precision_recall(true_positions, coords, diameters=diameters, threshold=0.5)
	# print('local_max', prec, rec)


	pred, df = dc.run_trackpy(label, diameter = dc.round_up_to_odd(params['r']*2))
	pred = find_positions(label, threshold=0)
	print(dc.round_up_to_odd(params['r']*2))
	print(pred.shape)
	# print(pred, true_positions)
	pred = exclude_borders(pred, canvas_size, pad=params['r']/4)
	print(true_positions.shape)
	print(pred.shape)

	prec, rec = dc.get_precision_recall(true_positions, pred, diameters=diameters, threshold=0.5)
	print('unet', prec, rec)


	# ap, precisions, recalls, thresholds = dc.average_precision(true_positions, tp_pred, diameters)

	# fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='tp on label', tag='o-', color='red')
	# plt.show()

	# x,y = dc.get_gr(true_positions, 50, 50, )
	# plt.plot(x, y, label=f'true n ={len(true_positions)}', color='black')
	# x,y = dc.get_gr(tp_pred, 50, 50, )
	# plt.plot(x, y, label=f'trackpy n ={len(tp_pred)}', color='grey')
	# # x,y = dc.get_gr(coords, 50, 50, )
	# # plt.plot(x, y, label=f'local_max n ={len(coords)}', color='red')
	# plt.legend()
	# plt.show()

	print(estimate_noise(canvas[0]))
	print("snr ", params['brightness']/estimate_noise(canvas[0]))

	dc.view(canvas, label=label, positions=tp_pred, )#true_positions=true_positions)

	# dc.view(canvas, label=label, true_positions=true_positions, )#true_positions=true_positions)