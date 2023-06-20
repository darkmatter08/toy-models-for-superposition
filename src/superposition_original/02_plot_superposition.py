from matplotlib import colors as mcolors
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from cycler import cycler

import torch
import numpy as np

from src.model_io import load_model
from src.superposition_original.data_structure.multi_model import \
    MultiModelConfig, \
    SuperPositionOriginal


def matplotlib_superposition(
    model_state_dict,
    model_config: MultiModelConfig,
    which=np.s_[:]
):
    n_feat = model_config.n_feat
    n_model = model_config.n_model
    # n_hidden = model_config.n_hidden
    

    feat_prob_3t = model_config.feat_prob_3t
    # (n_model, 1, 1)

    w_3t = model_state_dict['w_3t'].detach()
    # (n_model, n_feat, n_hidden)

    n_hidden_norm_3t = torch.linalg.norm(
        w_3t, 
        2, 
        dim=-1, 
        keepdim=True
    )
    # vec_norm = (n_model, n_feat, 1)

    w_unit_3t = w_3t / (1e-5 + n_hidden_norm_3t)

    w_unit_dot_w_3t = torch.einsum(
        'ifh,igh->ifg', 
        w_unit_3t, 
        w_3t
    )

    w_unit_dot_w_3t[
        :, 
        torch.arange(n_feat), 
        torch.arange(n_feat)
    ] = 0

    w_unit_dot_w_norm_2t = torch.linalg.norm(
        w_unit_dot_w_3t, 
        dim=-1
    )

    # avg_w_unit_dot_w_2t = (
    #     w_unit_dot_w_3t**2 * feat_prob_3t[:, None, :]
    # ).sum(-1).cpu()

    n_hidden_norm_2t = torch.linalg.norm(
        w_3t, 
        2, 
        dim=-1
    )
    # vec_norm = (n_model, n_feat), no keepdim for last dim 

    w_n_wT_3t = torch.einsum(
        'ifh,igh->ifg', 
        w_3t, 
        w_3t
    ).cpu()

    b_3t = model_state_dict['b_2t'].detach()
    # (n_model, 1, n_feat)
    b_3t = b_3t.view(n_model, n_feat, 1)

    which_instances = np.arange(n_model)[which]

    for inst in which_instances:
        # weights + bias
        fig = plt.figure(figsize=(12,12))
        gs = fig.add_gridspec(
            1,2, # row, col 
            width_ratios=(100,1),
            wspace=0.05
        )
        ax_w = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])

        w_n_wT_2t = w_n_wT_3t[inst].numpy()
        # (n_feat, n_feat)

        ax_w.imshow(
            w_n_wT_2t,
            cmap='RdBu_r',
            vmin=-1.3,
            vmax=1.3
        )

        feat_prob = feat_prob_3t[inst][0][0].item()
        sparsity = round(1-feat_prob,4)

        ax_w.set_title(f'Model {inst}, sparsity:{sparsity}')

        b_2t = b_3t[inst].numpy()
        # (n_feat, 1)
        ax_b.imshow(
            b_2t,
            cmap='RdBu_r',
            vmin=-1.3,
            vmax=1.3
        )

        ax_b.set_xticks([])
        ax_b.set_yticks([])

        plt.savefig(
            f'src/superposition_original/viz/02_{inst}_WnB.png'
        )

        fig, ax = plt.subplots(
                1,1, 
                figsize=(5, 20),
            )

        w_unit_dot_w_norm_1t = w_unit_dot_w_norm_2t[inst]
        horizontal_bar_color_list = [
            'black' if cross_dot_prod <= 0.20 else 'blue' 
            for cross_dot_prod in w_unit_dot_w_norm_1t
        ] 

        n_hidden_norm_1t = n_hidden_norm_2t[inst]
        ax.barh(
            np.arange(n_feat),
            n_hidden_norm_1t,
            align='center',
            color=horizontal_bar_color_list
            )

        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xticks([])

        ax.set_title(f'Model {inst}, sparsity:{sparsity}')

        plt.savefig(
            f'src/superposition_original/viz/03_{inst}_feat_superposition.png'
        )


if __name__ == "__main__":
    # plot_stah_diagram()
    all_models = SuperPositionOriginal()
    medium_model = all_models.medium_model
    medium_model_state_dict = load_model(
        file_path_str=medium_model.weights_file_path_str
    )

    which = np.s_[::2]
    matplotlib_superposition(
        model_state_dict=medium_model_state_dict,
        model_config=medium_model,
        which=which
    )