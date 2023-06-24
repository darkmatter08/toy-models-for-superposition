import torch
import matplotlib.pyplot as plt
import numpy as np

from src.superposition_original.data_structure.multi_model import \
    MultiModelConfig

def plot_dimensionality(
    model_state_dict,
    model_config: MultiModelConfig
):
    """

    dim_to_feat w_i = 1/2 means 1 dimension represent 2 features

    dim_to_feat w_i = w_i_vec_norm_square / sum_dot(w_i, w_j)

    w = (n_model, n_feat, n_hidden)
    w_i = (n_model, i, n_hidden)
    w_i_vect_norm = norm over n_hidden = (n_model, i, 1) 
    w_i_vect_norm_square = w_i_vect_norm_square

    sum_dot(w_i, w_j) = sum over all i, j feature pairs, dot(w_i, w_j)

    """

    w_3t = model_state_dict['w_3t']
    # w_3t = (n_model, n_feat, n_hidden)

    n_hidden_norm_3t = torch.linalg.norm(
        w_3t, 
        2, 
        dim=-1, 
        keepdim=True
    )
    # vec_norm = (n_model, n_feat, 1)

    w_unit_3t = w_3t / (1e-6 + n_hidden_norm_3t)

    w_unit_dot_w_3t = torch.einsum(
        'ifh,igh->ifg', 
        w_unit_3t, 
        w_3t
    )
    # w_unit_dot_w_3t = (n_model, n_unit_feat, n_feat)

    # n_feat = model_config.n_feat
    # w_unit_dot_w_3t[
    #     :, 
    #     torch.arange(n_feat), 
    #     torch.arange(n_feat)
    # ] = 0

    w_unit_dot_w_square_3t = w_unit_dot_w_3t ** 2
    sum_of_interference = w_unit_dot_w_square_3t.sum(-1)
    # sum_of_interference = (n_model, n_feat)

    n_hidden_norm_2t = n_hidden_norm_3t[:,:,0]

    feat_dim_2t = n_hidden_norm_2t ** 2 / sum_of_interference
    # feat_dim_2t = (n_model, n_feat)

    fig, ax = plt.subplots(
        1,1,
        figsize=(20,10)
    )


    ax.set_title(f'Const Feature Importance Model: Dimensionality of feature vs increasing sparsity')
    plot_horizontal_lines(ax)
    
    feat_prob_3t = model_config.feat_prob_3t
    feat_prob_1t = feat_prob_3t[:, 0, 0]

    one_div_feat_prob_1t = 1 / feat_prob_1t
    plot_vertical_lines(
        ax, 
        one_div_feat_prob_1t, 
        feat_prob_1t
    )

    n_model = model_config.n_model
    
    for i in range(n_model):
        feat_dim_1t = feat_dim_2t[i]
        # (n_feat)
        n_feat = feat_dim_1t.shape[0]

        if i < n_model - 1:
            # not last element
            dx = one_div_feat_prob_1t[i + 1] - one_div_feat_prob_1t[i]

        y = feat_dim_1t
        x = 1 / feat_prob_1t[i] * np.ones(n_feat) \
            + dx * np.random.uniform(-0.1, 0.1, n_feat)
        
        ax.scatter(
            y=y,
            x=x,
            marker="."
        )
    ax.set_yticks(
        np.arange(-0.1, 1.1, 0.1)
    )
    ax.set_xscale('log')
    
    plt.savefig(
        f'src/superposition_original/viz/04_plot_scatter.png'
    )


def plot_horizontal_lines(ax):
    for a, b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
        val = a / b
        ax.axhline(
            y=val,
            color="red",
            alpha=0.2,
        )
        ax.annotate(
            text=f'{a}/{b}',
            xy=(1.01, val),
            xycoords=("axes fraction", "data")
        )

    ax.axhline(
            y=0,
            color="red",
            alpha=0.2,
        )

    ax.annotate(
            text=f'0',
            xy=(1.01, 0),
            xycoords=("axes fraction", "data")
        )

    ax.axhline(
            y=1,
            color="red",
            alpha=0.2,
        )

    ax.annotate(
            text=f'1',
            xy=(1.01, 1),
            xycoords=("axes fraction", "data")
        )

    for a, b in [(5,6), (4,5), (3,4),(3,8), (3,12), (3,20)]: 
        val = a / b
        ax.axhline(
            y=val,
            color="blue",
            alpha=0.2,
        )
        ax.annotate(
            text=f'{a}/{b}',
            xy=(1.02, val),
            xycoords=("axes fraction", "data")
        )


def plot_vertical_lines(
    ax, 
    one_div_feat_prob_1t,
    feat_prob_1t
):
    for i, one_div_feat_prob_1v in enumerate(one_div_feat_prob_1t): 
        one_div_feat_prob_round = round(one_div_feat_prob_1v.item(), 4)
        ax.axvline(
            x=one_div_feat_prob_round,
            color="blue",
            alpha=0.2,
        )
        feat_prob_text = round(feat_prob_1t[i].item(), 4)
        ax.annotate(
            text=f'{feat_prob_text}',
            xy=(one_div_feat_prob_round, -0.04),
            xycoords=("data", "axes fraction")
        )

if __name__ == "__main__":
    from src.superposition_original.data_structure.multi_model import \
        SuperPositionOriginal
    from src.model_io import \
        load_model

    all_models = SuperPositionOriginal()
    const_feat_weight_model = all_models.constant_feat_weight_model
    medium_model_state_dict = load_model(
        file_path_str=const_feat_weight_model.weights_file_path_str
    )

    plot_dimensionality(
        model_state_dict=medium_model_state_dict,
        model_config=const_feat_weight_model
    )