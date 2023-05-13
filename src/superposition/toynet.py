import torch
import torch.nn as nn
import torch.nn.functional as F
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
        n_in,
        n_mid
    ):

        super().__init__()
        
        w1_2T = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_in)
            )
        )
        w1_rand_norm_2T= nn.init.xavier_normal(
            w1_2T
        )

        self.w1_2T = w1_rand_norm_2T

        w2_2T = nn.parameter.Parameter(
            torch.empty(
                (n_in, n_mid)
            )
        )

        w2_rand_norm_2T = nn.init.xavier_normal_(
            w2_2T
        )
        self.w2_2T = w2_rand_norm_2T

        self.b_1T = nn.parameter.Parameter(
            torch.zeros(n_in)
        ) 

    def forward(self, x_2T):
        w1_x_2T = self.w1_2T @ x_2T 
        lin_x = self.w2_2T @ w1_x_2T + self.b_1T
        relu_x = self.relu(lin_x)
        return relu_x


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
        n_in,
        n_mid
    ):

        super().__init__()
        
        w1_2T = nn.parameter.Parameter(
            torch.empty(
                (n_mid, n_in)
            )
        )
        w1_rand_norm_2T= nn.init.xavier_normal(
            w1_2T
        )

        self.w1_2T = w1_rand_norm_2T

        self.b_1T = nn.parameter.Parameter(
            torch.zeros(n_in)
        ) 

    def forward(self, x_2T):
        w1_x_2T = self.w1_2T @ x_2T 
        lin_x = self.w1_2T.T @ w1_x_2T + self.b_1T
        relu_x = self.relu(lin_x)
        return relu_x   
    
    



    

