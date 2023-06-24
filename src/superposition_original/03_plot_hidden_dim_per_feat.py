import matplotlib.pyplot as plt
from cycler import cycler

import torch
import numpy as np

from src.model_io import load_model
from src.superposition_original.data_structure.multi_model import \
    MultiModelConfig, \
    SuperPositionOriginal


def plot_hidden_dims_per_feature(
    model_state_dict,
    model_config: MultiModelConfig
):
    """

    w_3t = (n_model, n_feat, n_hidden)
    w_2t = (n_feat, n_hidden)

    squeezing n_feat into n_hidden
    if successful, norm of (n_feat_1, n_hidden) = sum n_hidden square > 0 

    so (Frobenius Norm)**2 of w_2t = estimate of how many n_feat squeeze into n_hidden

    """
    w_3t = model_state_dict['w_3t'].detach()
    feat_prob_3t = model_config.feat_prob_3t

    n_hidden = model_config.n_hidden

    x = 1 / feat_prob_3t[:, 0, 0]

    frobenius_norm = torch.linalg.matrix_norm(
        w_3t,
        'fro',
    )

    successful_num_feature = frobenius_norm ** 2
    y = n_hidden / successful_num_feature

    fig, ax = plt.subplots(
        1,1,
        figsize=(20,10)
    )

    ax.set_title(f'Const Feature Importance Model: N hidden dim / successful_embedded_num_features')

    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_xscale('log') 
    ax.plot(
        x,
        y,
        marker='o'
    )

    ax.grid()

    plt.savefig(
        f'src/superposition_original/viz/03_hidden_dim_per_feat_vs_sparsity.png'
        )
if __name__ == "__main__":

    all_models = SuperPositionOriginal()
    const_feat_weight_model = all_models.constant_feat_weight_model
    medium_model_state_dict = load_model(
        file_path_str=const_feat_weight_model.weights_file_path_str
    )

    plot_hidden_dims_per_feature(
        model_state_dict=medium_model_state_dict,
        model_config=const_feat_weight_model
    )