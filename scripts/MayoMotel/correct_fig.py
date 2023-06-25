import colloidoscope
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import copy

def objective(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def plot_curves(this_axs, i, x, y, color='red'):
    # curve fit
    popt, _ = curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c, d, e, f = popt
    # plot input vs output
    # plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), (max(x) - min(x))/10)
    # calculate the output for the range
    y_line = objective(x_line, a, b, c, d, e, f)
    # print(x_line)
    # create a line plot for the mapping function
    this_axs[i].plot(x_line, y_line, '-', color=color)



if __name__ == "__main__":
    # df_path = "scripts/MayoMotel/unet_tp_df.html"
    df_path = "scripts/MayoMotel/unet_tp_df.html"
    
    
    losses = pd.read_html(df_path, index_col=0)[0]
    print(losses)
    
    # import pdb; pdb.set_trace()
    
    plot_params = ['volfrac', 'snr', 'cnr', 'particle_size', 'brightness', 'r']
    titles		= ['Density $\phi$', 'SNR', 'CNR', 'Size ($\mu m$)', '$f_\mu$ (0-255)', 'Radius (pxls)']

    fig,axs = plt.subplots(2,len(plot_params),  sharey=True)
    plt.tight_layout(pad=0)

    this_axs = axs[0]
    for i, p in enumerate(plot_params):
        this_df = losses[losses['type'].isin([p])]
        x = copy.deepcopy(this_df[p].values)
        y = copy.deepcopy(this_df['tp_precision'].values)
        this_axs[i].scatter(x=p, 		y = 'tp_precision', data=this_df, s=7.0, color='black', marker='<', alpha=0.5)
        plot_curves(this_axs, i, x, y, color='black')
        x = copy.deepcopy(this_df[p].values)
        y = copy.deepcopy(this_df['precision'].values)
        this_axs[i].scatter(x=p, 		y = 'precision', 	data=this_df, s=7.0, color='red', marker='>', alpha=0.5)
        plot_curves(this_axs, i, x, y, color='red')
        this_axs[i].set_xticks([])
        if i == 0: 
            this_axs[i].set_ylabel("Precision", fontsize='large')
            this_axs[i].set_yticks([0,0.25,0.5,0.75,1])
            this_axs[i].set_ylim(-0.1,1.1)
            this_axs[i].legend(["TP", "TP fit", "U-net", "U-net fit"])

    this_axs = axs[1]
    for i, p in enumerate(plot_params):
        this_df = losses[losses['type'].isin([p])]
        x = copy.deepcopy(this_df[p].values)
        y = copy.deepcopy(this_df['tp_recall'].values)
        this_axs[i].scatter(x=p, 		y = 'tp_recall', data=this_df, s=7.0, color='black', marker='<', alpha=0.5)
        plot_curves(this_axs, i, x, y, color='black')
        x = copy.deepcopy(this_df[p].values)
        y = copy.deepcopy(this_df['recall'].values)
        this_axs[i].scatter(x=p, 		y = 'recall', 	data=this_df, s=7.0, color='red', marker='>', alpha=0.5)
        plot_curves(this_axs, i, x, y, color='red')
        this_axs[i].set_xlabel(titles[i], fontsize='large')
        if i == 0: 
            this_axs[i].set_ylabel("Recall", fontsize='large')

    fig.set_figwidth(13)
    fig.suptitle("Precisions and Recalls", fontsize='xx-large', y=1.05)

    plt.savefig("output/figs/corrections/unet_tp_PR.png", bbox_inches='tight')