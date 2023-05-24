import torch
import numpy as np
import time
from tqdm import trange
import einops

from src.superposition_original.multi_model import \
    generate_data, \
    MultiModel

from src.model_io import \
    save_model

from pathlib import Path

def linear_lr(
    step:int, 
    total_step:int
):
    return (1 - (step/total_step))


def const_lr(*_):
    return 1.0

def cosine_decay_lr(
    step: int,
    total_step: int
):
    return np.cos(
        0.5 * np.pi * step / (total_step - 1)
    )

def optim_multi_model(
    multi_model,
    config_dict,
    render=False,
    n_data_point=1024,
    total_step=10_000,
    print_freq=100,
    lr=1e-3,
    lr_scale=const_lr,
    hooks=[]
):
    optim = torch.optim.AdamW(
        list(multi_model.parameters()),
        lr=lr
    )

    n_model = config_dict['n_model']
    n_data_point = config_dict['n_data_point']
    n_feat = config_dict['n_feat']
    feat_prob_t = config_dict['feat_prob_t']

    start = time.time()
    with trange(total_step) as range_step:
        for step in range_step:
            step_lr = lr * lr_scale(step, total_step)
            for group in optim.param_groups:
                group['lr'] = step_lr
            optim.zero_grad(set_to_none=True)

            x_3t = generate_data(
                n_model=n_model, 
                n_data_point=n_data_point,
                n_feat=n_feat,
                feat_prob_t=feat_prob_t
                )

            pred_3t = multi_model(x_3t)
            weighted_mean_sqr_error_3t = (
                config_dict['feat_weight_1t'] * 
                (x_3t.abs() - pred_3t.abs()) ** 2
            )

            loss = einops.reduce(
                weighted_mean_sqr_error_3t,
                'm d f -> d',
                'mean'
            ).sum()

            loss.backward()
            optim.step()

            if hooks:
                hook_data = dict(
                    model=multi_model,
                    step=step,
                    optim=optim,
                    w_mse=weighted_mean_sqr_error_3t,
                    loss=loss,
                    lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            
            if step % print_freq == 0 \
                or (step + 1 == total_step):
                range_step.set_postfix(
                    loss=loss.item() / n_model,
                    lr=step_lr
                )

    return multi_model


def train_multi_model():

    n_feat = 5
    n_hidden = 2
    n_model = 10

    save_multi_model_path_str = (
        'src/superposition_original/weights/multi_model.pth'
    )

    multi_model = MultiModel(
        n_feat=n_feat,
        n_model=n_model,
        n_hidden=n_hidden
    )

    feat_weight_t = (
        0.9 ** torch.arange(n_feat)
    )[None, :]
    # [0.9**0, 0.9**1, 0.9**2, ... ]
    # shape: (n_feat, 1)

    feat_prob_t= (
        20 ** -torch.linspace(0, 1, n_model)
    )[None, :]
    # [20 **-0, 20 **-1, ... ]
    # shape: (1, n_model)

    config_dict = {
        'n_model': n_model,
        'n_feat': n_feat,
        'n_hidden': n_hidden,

        'feat_weight_t': feat_weight_t,
        'feat_prob_t': feat_prob_t,
    }

    final_multi_model = optim_multi_model(
        multi_model=multi_model,
        config_dict=config_dict,
    )

    save_model(
        pt_model_state_dict=final_multi_model.state_dict(),
        file_path_str=save_multi_model_path_str,
    )


if __name__ == "__main__":
    train_multi_model()