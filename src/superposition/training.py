import torch
from torch import Tensor

from jaxtyping import Float 
from typing import Dict, Tuple, List, Any

from src.superposition.toynet import \
    OneWeightLinearNet, \
    TwoWeightLinearNet 

from src.superposition.generate_x import \
    generate_sparse_x_2t, \
    normalize_matrix

from safetensors import safe_open
from safetensors.torch import save_file

def train_toy_model(
    toy_model,

    save_folder_str: str,
    save_model_name:str,

    n_epoch:int,
    n_feat:int,
    n_data:int,
    sparsity: float,

    lr:float=0.001,
    weight_decay:float=0.01

) -> Tuple[List[float], Dict[str, Tensor]]:
    """

    train model and 
    returns training_lost_list, 

    """
    optim = torch.optim.AdamW(
        toy_model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epoch_loss_list = []
    # lost for each epoch

    weight_t = torch.tensor([
        0.7 ** i for i in range(n_feat)
    ]).to('cuda')

    for i in range(n_epoch):
        sparse_x_2t = generate_sparse_x_2t(
            n_feat=n_feat,
            n_data=n_data,
            frac_is_zero=sparsity
        )

        norm_x_2t = normalize_matrix(sparse_x_2t)

        norm_x_2t = norm_x_2t.to('cuda')
        actual_2t = norm_x_2t.to('cuda')

        optim.zero_grad()
        pred_2t: Float[Tensor, 'n_feat n_data'] = toy_model(
            norm_x_2t.float()
        )

        loss_t = mean_weighted_square_error_loss(
            pred_2t=pred_2t,
            actual_2t=actual_2t,
            weight_t=weight_t
        )

        loss_t.backward()
        optim.step()
        epoch_loss_list.append(loss_t.item())

        if i % 2000 == 0:
            print('epoch: ', i)
            print('loss: ', loss_t.item())

            file_path_str = f'{save_folder_str}/{save_model_name}_{i}.pt'

            save_model(
                pt_model_state_dict=toy_model.state_dict(),
                file_path_str=file_path_str
                )

    final_model_state_dict = toy_model.state_dict()
    final_file_path_str = f'{save_folder_str}/{save_model_name}_0.pt'
    save_model(
        pt_model_state_dict=final_model_state_dict,
        file_path_str=final_file_path_str
    )

    # save checkpoint
    return epoch_loss_list, final_model_state_dict

    
def mean_weighted_square_error_loss(
    pred_2t: Float[Tensor, 'n_feat n_data'],
    actual_2t: Float[Tensor, 'n_feat n_data'],
    weight_t: Float[Tensor, 'n_feat']
) -> Float[Tensor, '1']:
    sqr_err_loss_t = (actual_2t - pred_2t.abs()) ** 2
    w_se_loss_t = weight_t * sqr_err_loss_t
    # check above
    mw_se_loss_t = w_se_loss_t.mean()
    return mw_se_loss_t


def save_model(
    pt_model_state_dict,
    file_path_str:str,
):   
    save_file(
        pt_model_state_dict,
        file_path_str
    )


def load_model(
    file_path_str: str
) -> Dict[str, Any]:
    tensor_dict = {}

    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)

    return tensor_dict


def load_partial_model(
    file_path_str: str,
    model_key_list: List[str]
) -> Dict[str, Any]:
    tensor_dict = {}
    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in model_key_list:
            tensor_dict[key] = f.get_tensor(key)
    
    return tensor_dict
 