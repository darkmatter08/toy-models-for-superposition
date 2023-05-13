import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType 
# import torchvision


SEED = 0
torch.manual_seed(SEED)

class TwoWeightLinearNet(nn.Module):
    """
        y = RELU(w2 * w1 * x + b)

        x = (n_in, n_feat)
        w1 = (n_mid, n_in)
        w2 = (n_in, n_mid) 
        w2 * w1 * x = (n_in, n_feat)
        b = (n_in, *n_feat) 

    """
    def __init__(
        self,
        n_in: int,
        n_mid: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_in)
            )
        )
        w1_rand_norm_2t= nn.init.xavier_normal(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        w2_2t = nn.parameter.Parameter(
            torch.empty(
                (n_in, n_mid)
            )
        )

        w2_rand_norm_2t = nn.init.xavier_normal_(
            w2_2t
        )
        self.w2_2t = w2_rand_norm_2t

        self.b_1t = nn.parameter.Parameter(
            torch.zeros(n_in)
        ) 

    def forward(
        self, 
        x_2t: TensorType['n_in', 'n_feat']
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

        x = (n_in, n_feat)
        w1 = (n_mid, n_in)
        W1.T = (n_in, n_feat)
        b = (n_in, *n_feat) 

    """
    def __init__(
        self,
        n_in: int,
        n_mid: int
    ):

        super().__init__()
        
        w1_2t = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_in)
            )
        )
        w1_rand_norm_2t = nn.init.xavier_normal(
            w1_2t
        )

        self.w1_2t = w1_rand_norm_2t

        self.b_1t = nn.parameter.Parameter(
            torch.zeros(n_in)
        ) 

    def forward(
        self, 
        x_2t: TensorType["n_in", "n_feat"]
    ):
        w1_x_2t = self.w1_2t @ x_2t
        lin_x_2t = self.w1_2t.T @ w1_x_2t + self.b_1t
        relu_x_2t = self.relu(lin_x_2t)

        return relu_x_2t   
    




    

