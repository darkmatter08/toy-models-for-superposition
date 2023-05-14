import torch
from torch import Tensor
from jaxtyping import Float
from sklearn import preprocessing


def generate_sparse_x_2t(
    n_feat: int, 
    n_data: int,
    frac_is_zero: float
) -> Float[Tensor, 'n_feat n_data']:
    """

    generate sparse x_2t (n_feat, n_data)
        with 0 < frac_is_zero < 1
    frac_is_zero is sparsity

    """

    x_2t = torch.rand(
        (n_feat, n_data)
    )
    # all x_2t 0 < elements < 1 
    sparse_x_2t = torch.where(
        x_2t <= 1 - frac_is_zero,
        x_2t,
        torch.zeros(())
    )

    return sparse_x_2t


def normalize_matrix(
    x:Float[Tensor, 'n_feat n_data']
) -> Float[Tensor, 'n_feat n_data']:
    return torch.tensor(
        preprocessing.normalize(
            x, norm='l2'
        )
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sparse_x_2t = generate_sparse_x_2t(
            n_feat=10, 
            n_data=40, 
            frac_is_zero=0.2
        )
    # print(sparse_x_2t)
    plt.imshow(
        sparse_x_2t,
        cmap='seismic'
    )
    # red positive, white zero, blue negative
    plt.savefig('viz/01_sparse_x_2t.png')