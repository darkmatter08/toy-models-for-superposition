from matplotlib import colors as mcolors
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from cycler import cycler

import torch
import numpy as np

from src.model_io import load_model
from src.superposition_original.data_structure.multi_model import \
    MultiModelConfig, \
    SuperPositionOriginal

# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def plot_star_diagram(
):
    

    multi_model_state_dict = load_model(
        file_path_str="src/superposition_original/weights/multi_model.pth"
    )

    w_3t = multi_model_state_dict['w_3t'].detach()

    n_model, n_feat, n_hidden = w_3t.shape
    # n_model=10, n_feat=5, n_hidden=2

    feat_weight_np= (
        0.9 ** np.arange(n_feat)
    )[:]

    plt.rcParams["axes.prop_cycle"] = \
        cycler(
            "color",
            plt.cm.viridis(
                feat_weight_np                
            )
        )    
    
    plt.rcParams["figure.dpi"] = 200

    fig, ax_list = plt.subplots(
        1, n_model, 
        figsize=(2*n_model, 2)
    )

    for i, ax in enumerate(ax_list):
        w_model_2t = w_3t[i].detach().numpy()
        # w_model_2t: (feat, hidden) = (5, 2)
        color_list = [
            mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
        ]
        ax.scatter(
            w_model_2t[:, 0],
            # x coordinate
            w_model_2t[:, 1],
            # y coordinate
            c=color_list[0:n_feat]
        )

        ax.set_aspect('equal')

        feat_line_3np = np.stack(
            (   
                np.zeros_like(w_model_2t), 
                w_model_2t
            ),
            axis=1
        )

        # feat_line_3np: (feat, 2, hidden)
        #
        # feat_line_3np[0] 
        # = [[x,y],[0,0]] 
        # = line from (0,0) to (x,y)
        # where 0 is feature number 0, 
        #

        ax.add_collection(
            mc.LineCollection(
               feat_line_3np, 
               colors=color_list
            ) 
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z,z))
        ax.set_ylim((-z,z))

        ax.tick_params(
            left=True,
            labelleft=False,

            right=False,
            
            bottom=True,
            labelbottom=False
        )

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['bottom', 'left']:
            ax.spines[spine].set_position('center')

    plt.savefig(
        f'src/superposition_original/viz/01_scatter_plot.png'
    )


def matplotlib_superposition(
    model_state_dict,
    model_config: MultiModelConfig,
    which=np.s_[:]
):
    n_feat = model_config.n_feat
    n_model = model_config.n_model
    # n_hidden = model_config.n_hidden
    

    feat_prob_3t = model_config.feat_prob_3t
    # (n_model, 1, 1)

    w_3t = model_state_dict['w_3t'].detach()
    # (n_model, n_feat, n_hidden)

    n_hidden_norm_3t = torch.linalg.norm(
        w_3t, 
        2, 
        dim=-1, 
        keepdim=True
    )
    # vec_norm = (n_model, n_feat, 1)

    w_unit_3t = w_3t / (1e-5 + n_hidden_norm_3t)

    w_unit_dot_w_3t = torch.einsum(
        'ifh,igh->ifg', 
        w_unit_3t, 
        w_3t
    )

    w_unit_dot_w_3t[
        :, 
        torch.arange(n_feat), 
        torch.arange(n_feat)
    ] = 0

    w_unit_dot_w_norm_2t = torch.linalg.norm(
        w_unit_dot_w_3t, 
        dim=-1
    )

    # avg_w_unit_dot_w_2t = (
    #     w_unit_dot_w_3t**2 * feat_prob_3t[:, None, :]
    # ).sum(-1).cpu()

    n_hidden_norm_2t = torch.linalg.norm(
        w_3t, 
        2, 
        dim=-1
    )
    # vec_norm = (n_model, n_feat), no keepdim for last dim 

    w_n_wT_3t = torch.einsum(
        'ifh,igh->ifg', 
        w_3t, 
        w_3t
    ).cpu()

    b_3t = model_state_dict['b_2t'].detach()
    # (n_model, 1, n_feat)
    b_3t = b_3t.view(n_model, n_feat, 1)

    which_instances = np.arange(n_model)[which]

    for inst in which_instances:
        # weights + bias
        fig = plt.figure(figsize=(12,12))
        gs = fig.add_gridspec(
            1,2, # row, col 
            width_ratios=(100,1),
            wspace=0.05
        )
        ax_w = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])

        w_n_wT_2t = w_n_wT_3t[inst].numpy()
        # (n_feat, n_feat)

        ax_w.imshow(
            w_n_wT_2t,
            cmap='RdBu_r',
            vmin=-1.3,
            vmax=1.3
        )

        feat_prob = feat_prob_3t[inst][0][0].item()
        sparsity = round(1-feat_prob,4)

        ax_w.set_title(f'Model {inst}, sparsity:{sparsity}')

        b_2t = b_3t[inst].numpy()
        # (n_feat, 1)
        ax_b.imshow(
            b_2t,
            cmap='RdBu_r',
            vmin=-1.3,
            vmax=1.3
        )

        ax_b.set_xticks([])
        ax_b.set_yticks([])

        plt.savefig(
            f'src/superposition_original/viz/02_{inst}_WnB.png'
        )

        fig, ax = plt.subplots(
                1,1, 
                figsize=(5, 20),
            )

        w_unit_dot_w_norm_1t = w_unit_dot_w_norm_2t[inst]
        horizontal_bar_color_list = [
            'black' if cross_dot_prod <= 0.20 else 'blue' 
            for cross_dot_prod in w_unit_dot_w_norm_1t
        ] 

        n_hidden_norm_1t = n_hidden_norm_2t[inst]
        ax.barh(
            np.arange(n_feat),
            n_hidden_norm_1t,
            align='center',
            color=horizontal_bar_color_list
            )

        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xticks([])

        ax.set_title(f'Model {inst}, sparsity:{sparsity}')

        plt.savefig(
            f'src/superposition_original/viz/03_{inst}_feat_superposition.png'
        )


# def plotly_superposition(
#     model_state_dict,
#     model_config: MultiModelConfig,
#     which=np.s_[:]
# ):
#     n_feat = model_config.n_feat
#     feat_prob_3t = model_config.feat_prob_3t
#     n_model = model_config.n_model
#     n_hidden = model_config.n_hidden

#     W = model_state_dict.W.detach()
#     # (n_model, n_feat, n_hidden)
#     n_hidden_norm = torch.linalg.norm(W, 2, dim=-1, keepdim=True)
#     # vec_norm = (n_model, n_feat, 1)

#     W_norm = W / (1e-5 + n_hidden_norm)

#     interference = torch.einsum(
#         'ifh,igh->ifg', 
#         W_norm, 
#         W
#     )

#     interference[:, torch.arange(n_feat), torch.arange(n_feat)] = 0

#     polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
#     net_interference = (
#         interference**2 * feat_prob_3t[:, None, :]
#     ).sum(-1).cpu()

#     norms = torch.linalg.norm(W, 2, dim=-1).cpu()
#     # vec_norm = (n_model, n_feat), no keepdim for last dim 

#     WtW = torch.einsum(
#         'sih,soh->sio', 
#         W, 
#         W
#     ).cpu()

#     # width = weights[0].cpu()
#     # x = torch.cumsum(width+0.1, 0) - width[0]
#     x = torch.arange(n_feat)
#     width = 0.9

#     which_instances = np.arange(n_model)[which]
#     fig = make_subplots(
#         rows=len(which_instances),
#         cols=2,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         horizontal_spacing=0.1
#     )

#     for (row, inst) in enumerate(which_instances):
#         fig.add_trace(
#             go.Bar(x=x, 
#                 y=norms[inst],
#                 marker=dict(
#                     color=polysemanticity[inst],
#                     cmin=0,
#                     cmax=1
#                 ),
#                 width=width,
#             ),
#             row=1+row, col=1
#         )
#         data = WtW[inst].numpy()
#         fig.add_trace(
#             go.Image(
#                 z=plt.cm.coolwarm((1 + data)/2, bytes=True),
#                 colormodel='rgba256',
#                 customdata=data,
#                 hovertemplate='''\
#                 In: %{x}<br>
#                 Out: %{y}<br>
#                 Weight: %{customdata:0.2f}
#                 '''            
#             ),
#             row=1+row, col=2
#         )

#     fig.add_vline(
#         x=(x[n_hidden-1]+x[n_hidden])/2, 
#         line=dict(width=0.5),
#         col=1,
#     )
    
#     # fig.update_traces(marker_size=1)
#     fig.update_layout(showlegend=False, 
#                         width=600,
#                         height=100*len(which_instances),
#                         margin=dict(t=0, b=0)
#     )

#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)

#     return fig

if __name__ == "__main__":
    # plot_stah_diagram()
    all_models = SuperPositionOriginal()
    medium_model = all_models.medium_model
    medium_model_state_dict = load_model(
        file_path_str=medium_model.weights_file_path_str
    )

    which = np.s_[::2]
    matplotlib_superposition(
        model_state_dict=medium_model_state_dict,
        model_config=medium_model,
        which=which
    )