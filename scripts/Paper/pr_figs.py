
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





def main():
	name = "unet_tp_1160"
	path = f"output/Paper/{name}.html"
	losses = pd.read_html(path, index_col=0)[0]
	print(losses)
	# exit()

	plot_params = ['volfrac', 'snr', 'cnr', 'particle_size', 'brightness', 'r']
	titles		= ['Density $\phi$', 'SNR', 'CNR', 'Size ($\mu m$)', '$f_\mu$ (0-255)', 'Radius (pxls)']

	fig,axs = plt.subplots(2,len(plot_params),  sharey=True)
	plt.tight_layout(pad=0)

	this_axs = axs[0,:].flatten()
	for i, p in enumerate(plot_params):
		this_df = losses[losses['type'].isin([p])]
		this_axs[i].scatter(x=p, 		y = 'tp_precision', data=this_df, color='black', marker='<')
		this_axs[i].scatter(x=p, 		y = 'precision', 	data=this_df, color='red', marker='>')
		this_axs[i].set_xticks([])
		if i == 0: 
			this_axs[i].set_ylabel("Precision", fontsize='large')
			this_axs[i].set_yticks([0,0.25,0.5,0.75,1])
			this_axs[i].set_ylim(-0.1,1.1)
			this_axs[i].legend(["TP", "U-net"])

	this_axs = axs[1,:].flatten()
	for i, p in enumerate(plot_params):
		this_df = losses[losses['type'].isin([p])]
		this_axs[i].scatter(x=p, 		y = 'tp_recall', data=this_df, color='black', marker='<')
		this_axs[i].scatter(x=p, 		y = 'recall', 	data=this_df, color='red', marker='>')
		this_axs[i].set_xlabel(titles[i], fontsize='large')
		if i == 0: 
			this_axs[i].set_ylabel("Recall", fontsize='large')

	fig.set_figwidth(13)
	fig.suptitle("Precisions and Recalls", fontsize='xx-large', y=1.05)
	path = f"output/Paper/{name}.png"
	plt.savefig(path)

if __name__ == "__main__":
	main()