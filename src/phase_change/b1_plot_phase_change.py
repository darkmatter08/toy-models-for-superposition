import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

dim = 3

def plot_phase_change():

    folder_name='src/phase_change/results/'
    with open(f'{folder_name}weight_exp_range_1np.npy', 'rb') as f:
        weight_exp_range_1np = np.load(f)
    
    with open(f'{folder_name}sparsity_exp_range_1np.npy', 'rb') as f:
        sparsity_exp_range_1np = np.load(f)

    with open(f'{folder_name}four_model_loss_3np.npy', 'rb') as f:
        four_model_loss_3np = np.load(f)

    with open(f'{folder_name}model_min_loss_2np.npy', 'rb') as f:
        model_min_loss_2np = np.load(f)
        # (sparsity, weight)


    fig, ax = plt.subplots(
        1,1,   
        figsize=(12,12)
    )

    ax.set_title("4 Model Phase Transition: sparsity vs weight")
    model_name_list = [
        'unweighted_sparse_one_element',
        'weighted_sparse_one_element',
        'unweighted_sparse_superposition_two_element',
        'weighted_sparse_superposition_two_element'
    ]

    color_map_func = colors.ListedColormap([
        '#039',
        '#ffa500',
        '#25b',
        '#d44'
    ])

    ax.imshow(
        model_min_loss_2np,
        cmap=color_map_func
    )

    x_tick_list = [0, 0.5, 1]
    num_weight, = weight_exp_range_1np.shape
    weight_index_list = [
        max(int(x_tick * num_weight)-1, 0)
        for x_tick in x_tick_list
    ]

    weight_exp_list = [
        weight_exp_range_1np[weight_index]
        for weight_index in weight_index_list
    ]

    weight_label_list = [
        round(10 ** weight_exp,1)
        for weight_exp in weight_exp_list
    ]

    ax.set_xticks(
        weight_index_list
    )
    ax.set_xticklabels(weight_label_list)
    ax.set_xlabel('weight')

    y_tick_list = [0, 0.25, 0.5, 0.75, 1]
    num_sparsity, = sparsity_exp_range_1np.shape    
    sparsity_index_list = [
        max(int(y_tick * num_sparsity)-1, 0)
        for y_tick in y_tick_list
    ]
    sparsity_exp_list = [
        sparsity_exp_range_1np[sparsity_index]
        for sparsity_index in sparsity_index_list
    ]
    sparsity_label_list = [
        round(10 ** sparsity_exp, 4)
        for sparsity_exp in sparsity_exp_list
    ]

    ax.set_yticks(
        sparsity_index_list
    )
    ax.set_yticklabels(sparsity_label_list)
    ax.invert_yaxis()
    ax.set_ylabel('sparsity')

    legend_handle_list = [
        mpatches.Patch(
            color=color_map_func(m_index),
            label=model_name
        )
        for m_index, model_name in enumerate(model_name_list)
    ]
    ax.legend(
        handles=legend_handle_list,
        loc='lower right'
    )

    plt.savefig(
        'src/phase_change/viz/b1_phase_change.png'
    )


if __name__ == "__main__":
    plot_phase_change()