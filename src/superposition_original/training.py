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

from src.superposition_original.data_structure.multi_model import \
    MultiModelConfig, \
    SuperPositionOriginal


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
    model_config: MultiModelConfig,
    render=False,
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

    n_model = model_config.n_model
    n_data_point = model_config.n_data_point
    n_feat = model_config.n_feat

    feat_prob_3t = model_config.feat_prob_3t.to("cuda")
    feat_weight_3t = model_config.feat_weight_3t.to("cuda") 

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
                feat_prob_3t=feat_prob_3t
                )

            pred_3t = multi_model(x_3t)
            weighted_mean_sqr_error_3t = (
                feat_weight_3t * 
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


def train_multi_model(
    model_config: MultiModelConfig
):
    n_feat = model_config.n_feat
    n_hidden = model_config.n_hidden
    n_model = model_config.n_model

    multi_model = MultiModel(
        n_feat=n_feat,
        n_model=n_model,
        n_hidden=n_hidden
    )

    final_multi_model = optim_multi_model(
        multi_model=multi_model,
        model_config=model_config,
    )

    save_model(
        pt_model_state_dict=final_multi_model.state_dict(),
        file_path_str=model_config.weights_file_path_str
    )

if __name__ == "__main__":
    all_model = SuperPositionOriginal()
    small_two_hidden_model = all_model.small_two_hidden_model

    train_multi_model(
        model_config=small_two_hidden_model
    )