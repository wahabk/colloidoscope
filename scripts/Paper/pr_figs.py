import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def add_model(this_df, this_axs, i, p, y="tp_precision", marker="r"):
    x_values = this_df[p].values
    y_values = this_df[y].values

    # Ensure x_values are sorted along with their corresponding y_values
    sorted_indices = np.argsort(x_values)
    x_values_sorted = x_values[sorted_indices]
    y_values_sorted = y_values[sorted_indices]

    # Initial guess for the parameters
    initial_guess = [1, 1, 1]

    # Fit the model
    params, _ = curve_fit(quadratic_model, x_values_sorted, y_values_sorted, p0=initial_guess)

    # Generate a set of evenly spaced x values for a smoother curve
    smooth_x_values = np.linspace(x_values_sorted.min(), x_values_sorted.max(), 500)
    fitted_vals = quadratic_model(smooth_x_values, *params)
    
    this_axs[i].plot(smooth_x_values, fitted_vals, marker, label='Fitted model', linewidth=2.5)

def plot_pr(losses:pd.DataFrame, title="str"):
    plot_params = ['volfrac', 'snr', 'cnr', 'particle_size', 'brightness', 'r']
    titles		= ['Density $\phi$', 'SNR', 'CNR', 'Size ($\mu m$)', '$f_\mu$ (0-255)', 'Radius (pxls)']

    fig,axs = plt.subplots(2,len(plot_params),  sharey=True)
    plt.tight_layout(pad=0)

    this_axs = axs[0,:].flatten()
    for i, p in enumerate(plot_params):
        this_df = losses[losses['type'].isin([p])]

        add_model(this_df, this_axs, i, p, "tp_precision", 'k')
        add_model(this_df, this_axs, i, p, "precision", 'r')

        this_axs[i].scatter(x=p, y = 'precision', 	data=this_df, color='red', marker='>', alpha=0.5)
        this_axs[i].scatter(x=p, y = 'tp_precision', data=this_df, color='black', marker='<', alpha=0.5)
        this_axs[i].set_xticks([])

        if i == 0: 
            this_axs[i].set_ylabel("Precision", fontsize='large')
            this_axs[i].set_yticks([0,0.25,0.5,0.75,1])
            this_axs[i].set_ylim(-0.1,1.1)
            this_axs[i].legend(["TP", "U-net"])

    this_axs = axs[1,:].flatten()
    for i, p in enumerate(plot_params):
        this_df = losses[losses['type'].isin([p])]

        add_model(this_df, this_axs, i, p, "tp_recall", 'k')
        add_model(this_df, this_axs, i, p, "recall", 'r')


        this_axs[i].scatter(x=p, 		y = 'tp_recall', data=this_df, color='black', marker='<', alpha=0.5)
        this_axs[i].scatter(x=p, 		y = 'recall', 	data=this_df, color='red', marker='>', alpha=0.5)
        this_axs[i].set_xlabel(titles[i], fontsize='large')
        if i == 0: 
            this_axs[i].set_ylabel("Recall", fontsize='large')

    fig.set_figwidth(12)
    fig.set_figheight(4)
    fig.suptitle(title, fontsize='xx-large', y=1.05)
    return fig

def main():
    names = ["unet_tp_1160", "unet_log_1161", "unet_att_log_1170"]
    titles = ["Unet (TP)", "Unet (LOG)", "Attention Unet (LOG)"]
    for name, title in zip(names, titles):
        path = f"output/Paper/{name}.html"
        losses = pd.read_html(path, index_col=0)[0]
        print(losses)
        # exit()

        fig = plot_pr(losses, title)
        path = f"output/Paper/{name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.clf()

if __name__ == "__main__":
    main()