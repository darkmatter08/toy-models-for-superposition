import torch
from typing import Tuple
from torch import Tensor
from jaxtyping import Float 
from src.superposition.training import \
    load_model
import matplotlib.pyplot as plt
import numpy as np

def cal_interference(
    w_2t: Float[Tensor,'feat hidden']
) -> Tuple[
    Float[Tensor,'feat'],
    Float[Tensor, 'feat']
]:
    n_feat = w_2t.size(0)

    w_vec_norm_2t = torch.linalg.norm(
            w_2t, 
            ord=2, # Sqr Root(a sqr + b sqr) = L2 norm
            dim=1, # eliminate dim=1 (feat, hidden) => (feat, 1)
            keepdim=True # returns (feat,1) size
        )
    # w_vec_norm_2t = (feat,1) , 
    # every feat has a norm

    w_unit_2t = w_2t / (
        w_vec_norm_2t + 1e-5
    )


    w_unit_to_w_2t = torch.einsum(
        'fh,gh->fg', 
        w_unit_2t, 
        w_2t
    )
    # w_unit_to_w_2t.shape = (w_unit_vec_feat, w_feat)
    # w_unit_to_w_2t[unit_index] = [all pairs norm_L2(b) * cos0]
    # hat(a) dot b = norm_L2(b) cos 0

    # set Identity to zero
    w_unit_to_w_2t[
        torch.arange(n_feat),
        torch.arange(n_feat)
    ] = 0

    feat_cross_dot_prod_L2_norm_1t =\
        torch.linalg.norm(
            w_unit_to_w_2t,
            ord=2, # L2
            dim=1, # eliminate w_feat: (w_unit_vec_feat, w_feat) => (w_unit_vec_feat)
            keepdim=False
        )
    feat_vec_norm_1t = w_vec_norm_2t.squeeze(1)
    return feat_cross_dot_prod_L2_norm_1t, \
        feat_vec_norm_1t


def plot_interference_one_weight():

    sparse_list = [0, 0.7, 0.9, 0.99, 0.999]
    fig, ax = plt.subplots(
            1,5, 
            figsize=(10, 3),
        )

    for i, sparsity in enumerate(sparse_list):
        file_path_str = (
            'src/superposition/weights/'
            f'one_w/s_{sparsity}/one_w_net_0.pt'
        )

        model_dict = load_model(
            file_path_str=file_path_str
        )   

        w1_2t = model_dict['w1_2t']
        feat_cross_dot_product_L2_norm_1t, \
        feat_vec_norm_1t  = cal_interference(
            w1_2t
        )
        # interference_norm_2t = (feat, feat)

        horizontal_bar_color_list = [
            'black' if cross_dot_prod <= 0.20 else 'blue' 
            for cross_dot_prod in feat_cross_dot_product_L2_norm_1t 
        ] 

        n_feat = w1_2t.size(0)

        ax[i].barh(
            np.arange(n_feat),
            feat_vec_norm_1t,
            align='center',
            color=horizontal_bar_color_list
            )
        ax[i].set_yticks([])
        ax[i].invert_yaxis()
        ax[i].set_xticks([])
        ax[i].set_title(f'1-S={1-sparsity:0.03f}')

    plt.savefig(
        f'src/superposition/viz/04_all_superposition_one_weight.png'
    )


def plot_interference_two_weight():

    sparse_list = [0, 0.7, 0.9, 0.99, 0.999]
    fig, ax = plt.subplots(
            1,5, 
            figsize=(10, 3),
        )

    for i, sparsity in enumerate(sparse_list):
        file_path_str = (
            'src/superposition/weights/'
            f'two_w/s_{sparsity}/two_w_net_0.pt'
        )

        model_dict = load_model(
            file_path_str=file_path_str
        )   

        w1_2t = model_dict['w1_2t']
        feat_cross_dot_product_L2_norm_1t, \
        feat_vec_norm_1t  = cal_interference(
            w1_2t
        )
        # interference_norm_2t = (feat, feat)

        horizontal_bar_color_list = [
            'black' if cross_dot_prod <= 0.20 else 'blue' 
            for cross_dot_prod in feat_cross_dot_product_L2_norm_1t 
        ] 

        n_feat = w1_2t.size(0)

        ax[i].barh(
            np.arange(n_feat),
            feat_vec_norm_1t,
            align='center',
            color=horizontal_bar_color_list
            )
        ax[i].set_yticks([])
        ax[i].invert_yaxis()
        ax[i].set_xticks([])
        ax[i].set_title(f'1-S={1-sparsity:0.03f}')

    plt.savefig(
        f'src/superposition/viz/04_all_superposition_two_weight.png'
    )
if __name__ == "__main__":

    # plot_interference_one_weight()
    plot_interference_two_weight()