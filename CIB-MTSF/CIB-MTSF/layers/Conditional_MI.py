import torch
from torch import nn
import torch.nn.functional as F


class FeedNet(nn.Module):
    def __init__(self, in_dim, out_dim, type="mlp", n_layers=1, inner_dim=None, activaion=None, dropout=0.1):
        super(FeedNet, self).__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = in_dim if i == 0 else inner_dim[i - 1]
            layer_out = out_dim if i == n_layers - 1 else inner_dim[i]
            if type == "mlp":
                self.layers.append(nn.Linear(layer_in, layer_out))
            else:
                raise Exception("KeyError: Feedward Net keyword error. Please use word in ['mlp']")
            if i != n_layers - 1 and activaion is not None:
                self.layers.append(activaion)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x



def compute_kl_divergence(mu, logvar):

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + 1e-10, dim=-1)


    kl_div = kl_div.mean()

    return kl_div


