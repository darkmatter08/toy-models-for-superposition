from pydantic import BaseModel
import torch

class MultiModelConfig(BaseModel):
    n_model: int
    n_data_point: int
    n_feat: int
    n_hidden: int

    feat_weight_3t: torch.Tensor  
    # feat_weight_3t = ( 0.9 ** torch.arange(n_feat) )[None, None, :]
    # [0.9**0, 0.9**1, 0.9**2, ... ]
    # shape: (1, 1, n_feat)
    # to match (model, data_point, feat)

    feat_prob_3t: torch.Tensor
    # feat_prob_3t = ( 20 ** -torch.linspace(0, 1, n_model))[:, None, None]
    # feat_prob = 1 - Sparsity
    # the probability of a feature being non zero 

    weights_file_path_str: str

class SuperPositionOriginal(BaseModel):
    small_two_hidden_model=MultiModelConfig(
        n_model = 10,
        n_data_point = 1024,
        n_feat = 5,
        n_hidden = 2,

        feat_weight_3t=( 0.9 ** torch.arange(5))[None, None, :],
        # feat_weight_3t=( 0.9 ** torch.arange(n_feat))[None, None, :],

        feat_prob_3t= ( 20 ** -torch.linspace(0, 1, 10))[:, None, None],
        # feat_prob_3t= ( 20 ** -torch.linspace(0, 1, n_model))[:, None, None],

        weights_file_path_str="src/superposition_original/weights/small_5_to_2_model.safetensors"
    )
    
    medium_model=MultiModelConfig(
        n_model = 20,
        n_data_point = 1024,
        n_feat = 100,
        n_hidden = 20,

        feat_weight_3t=( 100 ** -torch.linspace(0, 1, 100))[None, None, :],
        # feat_weight_3t=( 100 ** torch.linspace(0, 1, n_feat))[None, None, :],

        feat_prob_3t= ( 20 ** -torch.linspace(0, 1, 10))[:, None, None],
        # feat_prob_3t= ( 20 ** -torch.linspace(0, 1, n_model))[:, None, None],

        weights_file_path_str="src/superposition_original/weights/medium_100_to_20_model.safetensors"
    )

    constant_feat_weight_model=MultiModelConfig(
        n_model = 20,
        n_data_point = 1024,
        n_feat = 200,
        n_hidden = 20,

        feat_weight_3t=( 1 ** torch.arange(100))[None, None, :],
        # feat_weight_3t=( 1 * torch.arange(100))[None, None, :],
        # equal weight model
        feat_prob_3t= ( 20 ** -torch.linspace(0, 1, 10))[:, None, None],
        # feat_prob_3t= ( 20 ** -torch.linspace(0, 1, n_model))[:, None, None],

        weights_file_path_str="src/superposition_original/weights/constant_feat_weight_model.safetensors"
    )
