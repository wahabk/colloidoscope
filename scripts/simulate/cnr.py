from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
from scipy import ndimage
from skimage import io
import trackpy as tp
from numba import njit
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
# from torchvision.transforms.functional import equalize
import seaborn as sns
import napari
from scipy.signal import convolve, convolve2d

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside

def read_real_examples():

	d = {}

	d['abraham'] = {}
	d['abraham']['diameter'] = [13,11,11]
	d['abraham']['array'] = io.imread('examples/Data/abraham.tiff')
	d['emily'] = {}
	d['emily']['diameter'] = [15,9,9]
	d['emily']['array'] = io.imread('examples/Data/emily.tiff')
	d['katherine'] = {}
	d['katherine']['diameter'] = [7,7,7]
	d['katherine']['array'] = io.imread('examples/Data/katherine.tiff')
	d['levke'] = {}
	d['levke']['diameter'] = [15,11,11]
	d['levke']['array'] = io.imread('examples/Data/levke.tiff')
	d['james'] = {}
	d['james']['diameter'] = [17,15,15]
	d['james']['array'] = io.imread('examples/Data/james.tiff')
	d['jamesdecon'] = {}
	d['jamesdecon']['diameter'] = [17,15,15]
	d['jamesdecon']['array'] = io.imread('examples/Data/jamesdecon.tiff')

	return d

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def run_trackpy(self, array, diameter=5, *args, **kwargs) -> np.ndarray: #, pd.DataFrame]
		df = tp.locate(array, diameter=diameter, *args, **kwargs)
		f = list(zip(df['z'], df['y'], df['x']))
		tp_predictions = np.array(f, dtype='float32')

		return tp_predictions, df

def make_mask(canvas_size, positions, diameter):
	pass

@njit()
def draw_mask_slice(args):
	# extract args
	s, z, r, centers = args
	#initiate new slice to be drawn
	new_slice = s
	#for each sphere check if this pixel is inside it
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				# euclidean distance
				dist = math.sqrt((z - cz)**2 + (i - cy)**2 + (j - cx)**2)
				
				if dist <= r:
					new_slice[i,j] = 1
	return new_slice

def make_mask(shape, centers, diameter, num_workers=10):
	r = np.min(diameter)/2
	canvas = np.zeros(array.shape)
	args = [(s, z, r, centers) for z, s in enumerate(canvas)]

	mask = []
	with tqdm(total=len(args)) as pbar:
		with ProcessPoolExecutor(max_workers=num_workers) as pool:
			for i in pool.map(draw_mask_slice, args):
				mask.append(i)
				pbar.update(1)

	mask = list(mask)
	mask = np.array(mask, dtype='uint8')
	return mask

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()

	for (name, d) in real_dict.items():

		print(name)
		array = d['array']

		if name == 'emily': 
			array = dc.crop3d(array, (128,128,128))

		print(array.shape)

		positions, df = dc.run_trackpy(array, diameter = d['diameter'])
		print(positions.shape)
		dc.view(array, positions=positions)

		x, y = dc.get_gr(positions, 100, 100)
		plt.plot(x, y)
		plt.show()

		print('n_particles', positions.shape[0])
		# print(df)


		

		# label to find CNR of foreground and backgreound
		mask = make_mask(array.shape, positions, diameter=d['diameter'])
		# dc.view(array, positions=positions, label=mask) # check tp preds

		foreground = array[mask == 1]
		background = array[mask == 0]
		print(np.shape(foreground))
		print(np.shape(background))
		f_mean = foreground.mean()
		f_std = foreground.std()
		b_mean = background.mean()
		b_std = background.mean()

		cnr = abs(f_mean - b_mean) / b_std
		print(cnr, f_std)

		print('cnr, f_mean, f_std, b_mean, b_std')
		print(cnr, f_mean, f_std, b_mean, b_std)

		#find SNR
		noise = np.array([estimate_noise(s) for s in array])
		snr = f_mean / noise.mean()


		print("NOISE", name, noise.mean(), noise.std())
		print("SNR", snr)

		plt.hist(x=[foreground, background], bins=50, label=[f'foreground {f_mean:.2f}±{f_std:.2f}', f'background {b_mean:.2f}±{b_std:.2f}'], density=True)
		plt.title(f'{name}, cnr = {cnr:.2f}, snr = {snr:.2f}')
		plt.legend()
		plt.show()
		# plt.savefig(f'output/figs/cnr/{name}_cnr.png')
		plt.clf()



		# exit()

		# point = [63, 85, 106]
		# particle = dc.crop3d(array, roiSize=[2,24,24], center=point)[1]
		# print(particle.shape)

		# dc.view(particle)

		# rows = [particle[:,i] for i in range(8,15)]

		# [plt.plot(row, label=str(i)) for i, row in enumerate(rows)]
		# plt.xlim((0,23))
		# plt.legend()
		# plt.show()


