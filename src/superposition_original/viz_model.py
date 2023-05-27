from matplotlib import colors as mcolors
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from cycler import cycler

import torch
import numpy as np

from src.model_io import load_model

def plot_star_diagram(
):
    

    multi_model_state_dict = load_model(
        file_path_str="src/superposition_original/weights/multi_model.pth"
    )

    w_3t = multi_model_state_dict['w_3t'].detach()

    n_model, n_feat, n_hidden = w_3t.shape
    # n_model=10, n_feat=5, n_hidden=2

    feat_weight_np= (
        0.9 ** np.arange(n_feat)
    )[:]

    plt.rcParams["axes.prop_cycle"] = \
        cycler(
            "color",
            plt.cm.viridis(
                feat_weight_np                
            )
        )    
    
    plt.rcParams["figure.dpi"] = 200

    fig, ax_list = plt.subplots(
        1, n_model, 
        figsize=(2*n_model, 2)
    )

    for i, ax in enumerate(ax_list):
        w_model_2t = w_3t[i].detach().numpy()
        # w_model_2t: (feat, hidden) = (5, 2)
        color_list = [
            mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
        ]
        ax.scatter(
            w_model_2t[:, 0],
            w_model_2t[:, 1],
            c=color_list[0:n_feat]
        )

        ax.set_aspect('equal')

        stacked_np = np.stack(
            (np.zeros_like(w_model_2t), 
             w_model_2t),
            axis=1
        )
        # stacked_np: (2, feat, hidden)
        # stacked_np[0] = zeros_like()
        # stacked_np[1] = w_model_2t

        ax.add_collection(
            mc.LineCollection(
               stacked_np, 
               colors=color_list
            ) 
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z,z))
        ax.set_ylim((-z,z))

        ax.tick_params(
            left=True,
            labelleft=False,

            right=False,
            
            bottom=True,
            labelbottom=False
        )

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['bottom', 'left']:
            ax.spines[spine].set_position('center')

    plt.savefig(
        f'src/superposition_original/viz/01_scatter_plot.png'
    )

if __name__ == "__main__":
    plot_star_diagram()