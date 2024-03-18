from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import math 

from scripts.Paper.real_cnr import read_real_examples

def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

def signaltonoise2(array):
    signal = array.mean()
    noise = array.std()
    return signal/noise

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()
 
	real_len = len(real_dict)
	fig, axs = plt.subplots(1, real_len)
	plt.tight_layout(pad=0)
 
	for i, (name, d) in enumerate(real_dict.items()):
		array = d['array']
		array = np.array((array/array.max())*255,dtype='uint8')
		print(name, array.mean(), array.min(), array.max())
		snr = signaltonoise2(array)

		array = array.flatten()

		axs[i].hist(array, bins=50)
		axs[i].set_xlabel('Brightness', fontsize=12)
		axs[i].set_ylabel('Frequency', fontsize=12)
		axs[i].set_yticks([])
		axs[i].set_xlim(0,255)
		axs[i].set_title(f'SNR = {snr:.2f}', fontsize=12)
	fig.set_figwidth(12)
	fig.set_figheight(6)
	plt.savefig("output/Paper/real_snr.png", bbox_inches="tight")



	