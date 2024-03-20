from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from scripts.Paper.real_cnr import read_real_examples, GFP_CMAP


if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()
 
	fig, axs = plt.subplots(2,len(real_dict),sharex=True,sharey=True)
	fig.set_size_inches(12, 6)

	plt.tight_layout()

	for i, (name, d) in enumerate(real_dict.items()):
		array = d['array']
		array = dc.crop3d(array, (100,100,100))
		array = ndimage.zoom(array,2)
		print(name, array.shape, i)
  
		projection = np.max(array, axis=0)
		axs[0,i].imshow(projection, cmap=GFP_CMAP)
		axs[0,i].set_title(name,fontsize=11)
		axs[0,i].set_xticks([])
		axs[0,i].set_yticks([])
		if i == 0: axs[0,i].set_xlabel("X")
		if i == 0: axs[0,i].set_ylabel("Y")

		projection = np.max(array, axis=1)
		axs[1,i].imshow(projection, cmap=GFP_CMAP)
		axs[1,i].set_title("")
		axs[1,i].set_xticks([])
		axs[1,i].set_yticks([])
		if i == 0: axs[1,i].set_xlabel("X")
		if i == 0: axs[1,i].set_ylabel("Z")
	fig.set_figwidth(12)
	fig.set_figheight(4)
	plt.savefig("output/Paper/real.png", bbox_inches="tight")
  
