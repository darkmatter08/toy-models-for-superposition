import torch
from typing import List
from pathlib import Path

from src.superposition.training import \
    train_toy_model

from src.superposition.toynet import \
    OneWeightLinearNet, \
    TwoWeightLinearNet


def test_train_one_weight_linear_net():
    n_data = 1024
    n_feat = 20
    n_hidden = 5

    one_w_net_folder_path_str = 'src/superposition/weights/one_w'
    one_w_net_name = 'one_w_net'

    # y = RELU(w1.T * w1 * x + b)
    one_w_net = OneWeightLinearNet(
        n_feat=n_feat,
        n_hidden=n_hidden
    )

    # two_w_net_folder_path_str = 'weights/two_w/'
    # two_w_net_name = 'two_w_net'

    # # y = RELU(w2 * w1 * x + b)
    # two_w_net = TwoWeightLinearNet(
    #     n_feat=n_feat,
    #     n_mid=n_mid
    # )

    # test run 
    n_epoch = 1000
    sparsity = 0

    SEED = 0
    torch.manual_seed(SEED)

    train_lost_list, final_model_state_dict = train_toy_model(
        toy_model=one_w_net,

        save_folder_str=one_w_net_folder_path_str,
        save_model_name=one_w_net_name,

        n_epoch=n_epoch,
        n_data=n_data,    
        n_feat=n_feat,
        sparsity=sparsity
    )

def test_train_two_weight_linear_net():
    n_data = 1024
    n_feat = 20
    n_hidden = 5

    two_w_net_folder_path_str = 'src/superposition/weights/two_w'
    two_w_net_name = 'two_w_net'

    # y = RELU(x * w1 * w2  + b)
    two_w_net = TwoWeightLinearNet(
        n_feat=n_feat,
        n_hidden=n_hidden
    )

    # test run 
    n_epoch = 1000
    sparsity = 0

    SEED = 0
    torch.manual_seed(SEED)

    train_lost_list, final_model_state_dict = train_toy_model(
        toy_model=two_w_net,

        save_folder_str=two_w_net_folder_path_str,
        save_model_name=two_w_net_name,

        n_epoch=n_epoch,
        n_data=n_data,    
        n_feat=n_feat,
        sparsity=sparsity
    )


def multi_sparsity_one_weight_linear_net(
    sparsity_list: List[float]
):
    n_data = 1024
    n_feat = 20
    n_hidden = 5

    two_w_net_name = 'one_w_net'

    # y = RELU(x * w1 * w2  + b)
    two_w_net = OneWeightLinearNet(
        n_feat=n_feat,
        n_hidden=n_hidden
    )

    # test run 
    n_epoch = 10_000

    SEED = 0
    torch.manual_seed(SEED)

    for sparsity in sparsity_list:

        two_w_net_folder_path_str = f'src/superposition/weights/one_w/s_{sparsity}'

        folder_path = Path(two_w_net_folder_path_str)
        if not folder_path.exists():
            folder_path.mkdir()


        train_lost_list, final_model_state_dict = train_toy_model(
            toy_model=two_w_net,

            save_folder_str=two_w_net_folder_path_str,
            save_model_name=two_w_net_name,

            n_epoch=n_epoch,
            n_data=n_data,    
            n_feat=n_feat,
            sparsity=sparsity
        )


def multi_sparsity_two_weight_linear_net(
    sparsity_list: List[float]
):
    n_data = 1024
    n_feat = 20
    n_hidden = 5


    # y = RELU(x * w1 * w2  + b)
    two_w_net = TwoWeightLinearNet(
        n_feat=n_feat,
        n_hidden=n_hidden
    )

    # test run 
    n_epoch = 10_000

    SEED = 0
    torch.manual_seed(SEED)

    for sparsity in sparsity_list:

        two_w_net_folder_path_str = f'src/superposition/weights/two_w/s_{sparsity}'

        folder_path = Path(two_w_net_folder_path_str)
        if not folder_path.exists():
            folder_path.mkdir()

        two_w_net_name = 'two_w_net'

        train_lost_list, final_model_state_dict = train_toy_model(
            toy_model=two_w_net,

            save_folder_str=two_w_net_folder_path_str,
            save_model_name=two_w_net_name,

            n_epoch=n_epoch,
            n_data=n_data,    
            n_feat=n_feat,
            sparsity=sparsity
        )


if __name__ == "__main__":
    # test_train_one_weight_linear_net()
    # test_train_two_weight_linear_net()
    sparsity_list = [
        0, 
        0.7,
        0.9,
        0.99,
        0.999
    ]

    print('one_weight_linear_net')
    multi_sparsity_one_weight_linear_net(
        sparsity_list=sparsity_list
    )

    # print('two_weight_linear_net')
    # multi_sparsity_two_weight_linear_net(
    #     sparsity_list=sparsity_list
    # )