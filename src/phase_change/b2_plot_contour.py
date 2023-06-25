import numpy as np
import matplotlib.pyplot as plt    
from mpl_toolkits import mplot3d

def plot_contour():
    folder_name='src/phase_change/results/'
    with open(f'{folder_name}four_model_loss_3np.npy', 'rb') as f:
        four_model_loss_3np = np.load(f)

    with open(f'{folder_name}weight_exp_range_1np.npy', 'rb') as f:
        weight_exp_range_1np = np.load(f)
    
    with open(f'{folder_name}sparsity_exp_range_1np.npy', 'rb') as f:
        sparsity_exp_range_1np = np.load(f)

    least_explicit_loss_2np = np.min(four_model_loss_3np, axis=2)
    log_least_explicit_loss_2np = np.log(least_explicit_loss_2np)

    level_1np = np.linspace(
        log_least_explicit_loss_2np.min(),
        log_least_explicit_loss_2np.max(),
        40
    )

    fig, ax = plt.subplots(
        figsize=(12,12)
    )

    ax.contour(
        weight_exp_range_1np,
        sparsity_exp_range_1np,
        log_least_explicit_loss_2np,

        levels=level_1np,
        cmap='binary'
    )

    plt.savefig(
        'src/phase_change/viz/b2_log_loss.png'
    )


def plot_3d_plot():
    folder_name='src/phase_change/results/'
    with open(f'{folder_name}four_model_loss_3np.npy', 'rb') as f:
        four_model_loss_3np = np.load(f)

    with open(f'{folder_name}weight_exp_range_1np.npy', 'rb') as f:
        weight_exp_range_1np = np.load(f)
    
    with open(f'{folder_name}sparsity_exp_range_1np.npy', 'rb') as f:
        sparsity_exp_range_1np = np.load(f)

    least_explicit_loss_2np = np.min(four_model_loss_3np, axis=2)
    log_least_explicit_loss_2np = np.log(least_explicit_loss_2np)

    plt.figure(figsize=(12,8))
    plt.title("Least loss amaong explicit models")

    ax = plt.axes(projection='3d')
    ax.set_xlabel("log10 sparsity")
    ax.set_ylabel("log10 weight")
    ax.view_init(50, 170)
    ax.plot_surface(
        sparsity_exp_range_1np,
        weight_exp_range_1np,
        log_least_explicit_loss_2np,
        linewidth=0,
        antialiased=False
    )
    ax.set_title('surface')

    plt.savefig(
        'src/phase_change/viz/b2_3d_contour.png'
    )


if __name__ == "__main__":
    plot_contour()
    plot_3d_plot()