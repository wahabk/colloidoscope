from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import io
import trackpy as tp
from numba import njit
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import convolve2d

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside

def read_real_examples():
	d = {}

	d["A - Silica (560nm 0.55Φ)"] = {}
	d["A - Silica (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["A - Silica (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["A - Silica (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/james.tiff')
	d["B - Silica Decon. (560nm 0.55Φ)"] = {}
	d["B - Silica Decon. (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["B - Silica Decon. (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["B - Silica Decon. (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/jamesdecon.tiff')
	d["C - PMMA (315nm 0.58Φ)"] = {}
	d["C - PMMA (315nm 0.58Φ)"]['diameter'] = [15,11,11]
	d["C - PMMA (315nm 0.58Φ)"]['volfrac'] = 0.58
	d["C - PMMA (315nm 0.58Φ)"]['array'] = io.imread('examples/Data/levke.tiff')
	d["D - Silica (1.2μm 0.2Φ)"] = {}
	d["D - Silica (1.2μm 0.2Φ)"]['diameter'] = 15
	d["D - Silica (1.2μm 0.2Φ)"]['volfrac'] = 0.2
	d["D - Silica (1.2μm 0.2Φ)"]['array'] = ndimage.zoom(io.imread('examples/Data/katherine.tiff'), 2)

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

def make_mask(array, centers, diameter, num_workers=16):
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
	# dataset_path = '/media/wahab/SeagateExp/Data/colloids/'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()
	fig, axs = plt.subplots(1, len(real_dict))
	fig.set_size_inches(12, 6)
	plt.tight_layout(pad=0)

	for i, (name, d) in enumerate(real_dict.items()):
		array = d['array']
		# array = dc.crop3d(array, (100,100,100))
		array = np.array((array/array.max())*255,dtype='uint8')
		print(name, array.shape)

		positions, df = dc.run_trackpy(array, diameter = d['diameter'])
		print(positions.shape)

		print('n_particles', positions.shape[0])
		
		# label to find CNR of foreground and backgreound
		mask = make_mask(array, positions, diameter=d['diameter'])

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

		axs[i].hist(x=[foreground, background], bins=50, label=[f'F={f_mean:.0f}±{f_std:.1f}', f'B={b_mean:.0f}±{b_std:.1f}'], density=True)
		axs[i].set_title(f'CNR = {cnr:.1f}, SNR = {snr:.1f}', fontsize=12)
		axs[i].set_xlabel("Brightness")
		axs[i].set_ylabel("Frequency")
		axs[i].set_yticks([])
		axs[i].legend(fontsize=8)
	fig.set_figwidth(12)
	fig.set_figheight(2)
	plt.savefig(f'output/figs/real_data/real_cnr.png', bbox_inches="tight")


