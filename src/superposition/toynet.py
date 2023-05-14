import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

# import torchvision


class TwoWeightLinearNet(nn.Module):
    """
        y = RELU(x * w1 * w2 + b)
            pytorch nn.Linear usually is
            y = x * A.T + b

        x = (n_data, n_feat)

        w1 = (n_feat, n_hidden)
        w2 = (n_hidden, n_feat) 

        w2 * w1 * x = (n_data, n_feat)

        b = (*n_data, n_feat) 

    """
    def __init__(
        self,
        n_feat: int,
        n_hidden: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_feat, n_hidden)
            )
        )
        w1_rand_norm_2t= nn.init.xavier_normal_(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        w2_2t = nn.parameter.Parameter(
            torch.empty(
                (n_hidden, n_feat)
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
        x_2t: Float[Tensor, 'n_data n_feat']
    ):
        # y = RELU(x * w1 * w2 + b)
        # y (n_data, n_feat)
        
        x_w1_2t =  x_2t @ self.w1_2t
        lin_x_2t = x_w1_2t @ self.w2_2t + self.b_1t
        relu_x_2t = self.relu(lin_x_2t)
        return relu_x_2t


class OneWeightLinearNet(nn.Module):
    """
        y = RELU(
            x * w1 * w1.T + b
        )

        x = (n_data, n_feat)

        w1 = (n_feat, n_hidden)
        w1.T = (n_hidden, n_feat)

        b = (*n_data, n_feat) 

    """
    def __init__(
        self,
        n_feat: int,
        n_hidden: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_feat, n_hidden)
            )
        )
        w1_rand_norm_2t = nn.init.xavier_normal_(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        self.b_1t = nn.parameter.Parameter(
            torch.zeros(n_feat)
        ) 

        self.relu = nn.ReLU()

    def forward(
        self, 
        x_2t: Float[Tensor, "n_data n_feat"]
    ):
        x_w1_2t = x_2t @ self.w1_2t
        transpose_w1_2t = x_w1_2t @ self.w1_2t.T + self.b_1t
        relu_x_2t = self.relu(transpose_w1_2t)

        return relu_x_2t   
    




    

