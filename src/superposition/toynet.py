import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

# import torchvision


class TwoWeightLinearNet(nn.Module):
    """
        y = RELU(w2 * w1 * x + b)

        x = (n_feat, n_data)
        w1 = (n_mid, n_feat)
        w2 = (n_feat, n_mid) 
        w2 * w1 * x = (n_feat, n_data)
        b = (n_feat, *n_data) 

    """
    def __init__(
        self,
        n_feat: int,
        n_mid: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_feat)
            )
        )
        w1_rand_norm_2t= nn.init.xavier_normal(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        w2_2t = nn.parameter.Parameter(
            torch.empty(
                (n_feat, n_mid)
            )
        )

        w2_rand_norm_2t = nn.init.xavier_normal_(
            w2_2t
        )
        self.w2_2t = w2_rand_norm_2t

        self.b_1t = nn.parameter.Parameter(
            torch.zeros(n_feat)
        ) 

        self.relu = nn.ReLU()

    def forward(
        self, 
        x_2t: Float[Tensor, 'n_feat n_data']
    ):
        w1_x_2t = self.w1_2t @ x_2t
        lin_x_2t = self.w2_2t @ w1_x_2t + self.b_1t
        relu_x_2t = self.relu(lin_x_2t)
        return relu_x_2t


class OneWeightLinearNet(nn.Module):
    """
        y = RELU(
            w1.Transpose * w1 * x + b
        )

        x = (n_feat, n_data)
        w1 = (n_mid, n_feat)
        W1.T = (n_feat, n_mid)
        b = (n_feat, *n_data) 

    """
    def __init__(
        self,
        n_feat: int,
        n_mid: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_feat)
            )
        )
        w1_rand_norm_2t = nn.init.xavier_normal(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        self.b_1t = nn.parameter.Parameter(
            torch.zeros(n_feat)
        ) 

        self.relu = nn.ReLU()

    def forward(
        self, 
        x_2t: Float[Tensor, "n_feat n_data"]
    ):
        w1_x_2t = self.w1_2t @ x_2t
        lin_x_2t = self.w1_2t.T @ w1_x_2t + self.b_1t
        relu_x_2t = self.relu(lin_x_2t)

        return relu_x_2t   
    




    

