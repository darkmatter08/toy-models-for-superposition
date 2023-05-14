import torch
from torch import Tensor
from jaxtyping import Float 
from typing import Any

from src.toynet import \
    OneWeightLinearNet, \
    TwoWeightLinearNet 

from src.generate_x import \
    generate_sparse_x_2t

from safetensors import safe_open
from safetensors.torch import safe_file

def train_toy_model(
    toy_model_name:str,
    toy_model,
    n_epochs:int,
    n_feat:int,
    n_data:int,
    sparsity: float
):
    optim = torch.optim.AdamW(
        toy_model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    train_loss_list = []

    weight_t = torch.tensor([
        0.7 ** i for i in range(n_feat)
    ]).to('cuda')

    for i in range(n_epochs):
        sparse_x_2t = generate_sparse_x_2t(
            n_feat=n_feat,
            n_data=n_data,
            frac_is_zero=sparsity
        )

        sparse_x_2t = sparse_x_2t.to('cuda')
        actual_2t = sparse_x_2t.to('cuda')

        optim.zero_grad()
        pred_2t: Float[Tensor, 'n_feat n_data'] = toy_model(
            sparse_x_2t.float()
        )

        loss_t = mean_weighted_square_error_loss(
            pred_2t=pred_2t,
            actual_2t=actual_2t,
            weight_t=weight_t
        )

        loss_t.backward()
        optim.step()
        train_loss_list.append(loss_t)

        if i % 500 == 0:
            print('epoch: ', i)
            print('loss: ', loss_t.item())

            file_path_str = f'weights/train/{toy_model_name}_{i}.pt'

            save_model(
                pt_model=toy_model.state_dict(),
                file_path_str=file_path_str
                )

    final_model_weights = toy_model.state_dict()
    final_file_path_str = f'weight/train/{toy_model_name}_0.pt'
    save_model(
        pt_model=final_model_weight,
        file_path_str=file_path_str
    )

    # save checkpoint
    return loss_t.item(), final_model_weights

    
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
) -> Dict[str, Any]
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
) -> Dict[str, Any]
    tensor_dict = {}
    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in model_key_list:
            tensor_dict[key] = f.get_tensor(key)
    
    return tensor_dict
 