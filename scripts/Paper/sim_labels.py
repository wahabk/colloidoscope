from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path



if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/data/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(128,128,128)
	label_size=(128,128,128)
	num_workers = 16

	params = dict(
		r=12,
		particle_size=0.25,
		snr=4,
		cnr=4,
		volfrac=0.2,
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

	heatmap_rs = ["radius", 4, 8, "seg-4", "seg-8"]

	ims = []
	for i, h in enumerate(heatmap_rs):
		print(h)
		canvas, label, final_centers, final_diameters = dc.simulate(
					canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
					params['snr'], diameters=diameters, make_label=True, heatmap_r=h, 
					num_workers=num_workers, psf_kernel=psf_kernel)

		if i == 0:
			im_projs = [canvas.max(axis=0), canvas.max(axis=1)]
			im_proj = np.concatenate(im_projs, axis=0)
			ims.append(im_proj)

		label = label*255
		lab_projs = [label.max(axis=0), label.max(axis=1)]
		lab_proj = np.concatenate(lab_projs, axis=0)
		# both = np.concatenate([im_proj, lab_proj], axis=0)
		ims.append(lab_proj)
	
	img = ims[0]
	labels = np.concatenate(ims[1:], axis=1)
	plt.imsave("output/Paper/sim_labels.png", labels, cmap='bone')
	plt.imsave("output/Paper/sim_labels_img.png", img, cmap='gist_heat')
	all_ims = np.concatenate([labels, img], axis=1)
	plt.imsave("output/Paper/sim_labels_all.png", all_ims)
