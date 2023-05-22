import matplotlib.pyplot as plt
from src.superposition.training import \
    load_model
from torch import nn
import numpy as np


def viz_one_weight():
    sparsity = 0.99
    file_path_str = (
        'src/superposition/weights/'
        f'one_w/s_{sparsity}/one_w_net_0.pt'
    )

    model_dict = load_model(
        file_path_str=file_path_str
    )

    w1 = model_dict['w1_2t']
    # for model_key in model_dict.keys():
    #     print(model_key)

    w1_n_w1_t = w1 @ w1.T
    print(w1_n_w1_t)
    plt.imshow(
        w1_n_w1_t,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/03_w1_{sparsity}.png'
    )

    lin_x = w1_n_w1_t + model_dict['b_1t']
    print(lin_x)
    plt.imshow(
        lin_x,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/03_lin_X_{sparsity}.png'
    )

    nn_relu = nn.ReLU()
    relu_x = nn_relu( lin_x )

    plt.imshow(
        relu_x,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/03_relu_X_{sparsity}.png'
    )


def viz_two_weight():
    sparsity = 0.99
    file_path_str = (
        'src/superposition/weights/'
        f'two_w/s_{sparsity}/two_w_net_0.pt'
    )

    model_dict = load_model(
        file_path_str=file_path_str
    )

    w1 = model_dict['w1_2t']
    w2 = model_dict['w2_2t']
    # for model_key in model_dict.keys():
    #     print(model_key)

    w1_n_w1_t = w1 @ w2
    print(w1_n_w1_t)
    plt.imshow(
        w1_n_w1_t,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/two_w/03_w2_{sparsity}.png'
    )

    lin_x = w1_n_w1_t + model_dict['b_1t']
    print(lin_x)
    plt.imshow(
        lin_x,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/03_lin_X_{sparsity}.png'
    )

    nn_relu = nn.ReLU()
    relu_x = nn_relu(lin_x)

    plt.imshow(
        relu_x,
        vmin=-1.3,
        vmax=1.3,
        cmap='RdBu_r'
    )

    plt.savefig(
        f'src/superposition/viz/03_relu_X_{sparsity}.png'
    )


def plot_all_sparse_plot():
    sparse_list = [0, 0.7, 0.9, 0.99, 0.999]

    fig, ax = plt.subplots(
        1, 10,
        figsize=(25, 4),
        gridspec_kw={
            'width_ratios': [20, 1] * 5
        }       
    )
    
    for i, sparsity in enumerate(sparse_list):
        file_path_str = (
            'src/superposition/weights/'
            f'one_w/s_{sparsity}/one_w_net_0.pt'
        )

        model_dict = load_model(
            file_path_str=file_path_str
        )   

        w1 = model_dict['w1_2t']
        # w1 = (20, 5)
        w1_dot_w1_T = w1 @ w1.T
        
        b = model_dict['b_1t']

        w1_fig_index = 2 * i 
        b_fig_index = w1_fig_index + 1

        im = ax[w1_fig_index].imshow(
            w1_dot_w1_T,
            vmin=-1.3,
            vmax=1.3,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_title(
            f'1-S={1-sparsity:.03f}'
        )

        b_max = np.abs(b).max() * 1.3
        ax[b_fig_index].imshow(
            b.reshape(-1,1),
            vmin=-b_max,
            vmax=b_max,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_xticks([])
        ax[w1_fig_index].set_yticks([])


        ax[b_fig_index].set_xticks([])
        ax[b_fig_index].set_yticks([])

    plt.savefig(
        f'src/superposition/viz/03_all_sparsity_one_weight.png'
    )




def plot_all_sparse_plot_two_weight():
    sparse_list = [0, 0.7, 0.9, 0.99, 0.999]

    fig, ax = plt.subplots(
        1, 10,
        figsize=(25, 4),
        gridspec_kw={
            'width_ratios': [20, 1] * 5
        }       
    )
    
    for i, sparsity in enumerate(sparse_list):
        file_path_str = (
            'src/superposition/weights/'
            f'two_w/s_{sparsity}/two_w_net_0.pt'
        )

        model_dict = load_model(
            file_path_str=file_path_str
        )   

        w1 = model_dict['w1_2t']
        w2 = model_dict['w2_2t']
        # w1 = (20, 5)
        w1_dot_w2 = w1 @ w2
        
        b = model_dict['b_1t']

        w1_fig_index = 2 * i 
        b_fig_index = w1_fig_index + 1

        im = ax[w1_fig_index].imshow(
            w1_dot_w2,
            vmin=-1.3,
            vmax=1.3,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_title(
            f'1-S={1-sparsity:.03f}'
        )

        b_max = np.abs(b).max() * 1.3
        ax[b_fig_index].imshow(
            b.reshape(-1,1),
            vmin=-b_max,
            vmax=b_max,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_xticks([])
        ax[w1_fig_index].set_yticks([])


        ax[b_fig_index].set_xticks([])
        ax[b_fig_index].set_yticks([])

    plt.savefig(
        f'src/superposition/viz/03_all_sparsity_two_weight.png'
    )
    
if __name__ == "__main__":
    
    plot_all_sparse_plot_two_weight()

