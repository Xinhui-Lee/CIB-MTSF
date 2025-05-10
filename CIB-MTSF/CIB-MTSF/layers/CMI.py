import torch
from torch import nn
import torch.nn.functional as F

class ConditionalMINE(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(ConditionalMINE, self).__init__()

        self.fc1 = nn.Linear(d_model * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, patch1, patch2, condition):
        x = torch.cat((patch1, patch2, condition), dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)


def mine_loss(T_real, T_fake):

    cmi_loss = torch.mean(T_real) - torch.logsumexp(T_fake, dim=0) + torch.log(
        torch.tensor(T_fake.size(0), dtype=T_fake.dtype))

    return cmi_loss



def generate_samples(mine_model, qz, patch_idx1, patch_idx2):

    patch1 = qz[:, :, :, patch_idx1]
    patch2 = qz[:, :, :, patch_idx2]


    if patch_idx1 == 0:

        condition = torch.zeros_like(patch1)
    else:

        previous_patches = qz[:, :, :, :patch_idx1]

        condition = torch.mean(previous_patches, dim=-1)


    T_real = mine_model(patch1, patch2, condition)


    patch2_shuffle = patch2[torch.randperm(patch2.size(0))]
    T_fake = mine_model(patch1, patch2_shuffle, condition)

    return T_real, T_fake
