
import math
import numpy as np
from phase_change.a1_loss_fn import \
    weighted_sparse_one_element_expected_loss, \
    unweighted_sparse_one_element_expected_loss, \
    weighted_sparse_two_element_superposition_expected_loss, \
    unweighted_sparse_two_element_superposition_expected_loss

"""

Simulate 
    last_weight from 1e-2 to 1e2
    sparsity from    1e-4 to 1

and plot which model

    unweighted element & sparse
    weighted element & sparse
    unweighted superposition & sparse
    weighted superposition & sparse

has lower loss 

Lower loss model will be the choosen model in the given
    (last_weight, sparsity) pair

This will identify boundaries for phase transitions

"""

def simulate_phase_change(
):

    sparsity_intervals = 200
    min_sparsity_exp = -3
    max_sparsity_exp = 0
    # min_sparsity = 1e-4
    # max_sparsity = 1

    sparsity_exp_range_1np = np.linspace(
        min_sparsity_exp,
        max_sparsity_exp,
        sparsity_intervals
    )

    weight_intervals = 200
    min_weight_exp = -2
    max_weight_exp = 2
    # min_weight = 1e-2
    # max_weight = 1e2

    weight_exp_range_1np = np.linspace(
        min_weight_exp,
        max_weight_exp,
        weight_intervals
    )

    four_model_loss_3np = np.zeros((
        sparsity_intervals,
        weight_intervals,
        4
    ))

    
    for i, sparsity_exp in enumerate(sparsity_exp_range_1np):
        sparsity = 10 ** sparsity_exp
        for j, weight_exp in enumerate(weight_exp_range_1np):
            weight = 10 ** weight_exp
            four_model_loss_3np[i][j][0] = unweighted_sparse_one_element_expected_loss(sparsity)
            four_model_loss_3np[i][j][1] = weighted_sparse_one_element_expected_loss(weight, sparsity)
            four_model_loss_3np[i][j][2] = unweighted_sparse_two_element_superposition_expected_loss(sparsity)
            four_model_loss_3np[i][j][3] = weighted_sparse_two_element_superposition_expected_loss(weight, sparsity)

    
    model_min_loss_2np = np.argmin(
        four_model_loss_3np,
        axis=2 # four models choose the min loss model
    )

    folder_name = 'src/phase_change/results/'
    with open(f'{folder_name}sparsity_exp_range_1np.npy', 'wb') as f:
        np.save(f, sparsity_exp_range_1np)

    with open(f'{folder_name}weight_exp_range_1np.npy', 'wb') as f:
        np.save(f, weight_exp_range_1np)
    
    with open(f'{folder_name}four_model_loss_3np.npy', 'wb') as f:
        np.save(f, four_model_loss_3np)

    with open(f'{folder_name}model_min_loss_2np.npy', 'wb') as f:
        np.save(f, model_min_loss_2np)




if __name__ == "__main__":
    simulate_phase_change()