__all__ = ['CIBMTSF_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

from layers.Conditional_MI import FeedNet, compute_kl_divergence
from layers.CMI import mine_loss, generate_samples, ConditionalMINE


# Cell
class CIBMTSF_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, batch_size: int = 128, klweight: float = 1e-4,
                 contraweight: float = 1e-4, cmiweight: float = 1e-2, temperature:float=0.1, hidden_dim:int=256, select_channel:int=6, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.patch_num = patch_num


        self.backbone = TSTiEncoder(c_in, flag=True, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)


        self.contrabackbone = TSTiEncoder(c_in, flag=False, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=1, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)


        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)


        self.d_model = d_model
        self.batch_size = batch_size


        self.current_epoch = -1
        self.mine_model = ConditionalMINE(d_model=d_model, hidden_dim=hidden_dim)


        self.c_anchor = c_in // 2
        self.bs_anchor = batch_size // 2
        self.temperature = temperature

        self.cont_sigma = 1.0
        self.cont_length_scale = 2.0

        # weight
        self.klweight = klweight
        self.contraweight = contraweight
        self.cmiweight = cmiweight

        self.select_channel = select_channel

    def cauchy_contrast(self, C, sigma, length_scale):

        xs = torch.arange(C)
        xs_in = torch.unsqueeze(xs, 0)
        xs_out = torch.unsqueeze(xs, 1)
        distance_matrix = (xs_in - xs_out) ** 2
        distance_matrix_scaled = distance_matrix / length_scale ** 2
        kernel_matrix = torch.divide(sigma ** 2, (distance_matrix_scaled + 1.))
        return kernel_matrix



    def contrastive_loss(self, qz, x, select_channel):

        qz_anchor = qz[self.bs_anchor, self.c_anchor, :, :]  # [16, 42]


        mask = torch.ones_like(x)
        mask[self.bs_anchor, self.c_anchor, :, :] = 0
        x_masked = x * mask
        qz_masked = self.contrabackbone(x_masked)  # [128, 7, 16, 42]
        qz_masked = qz_masked[self.bs_anchor, :, :, :]  # [7, 16, 42]

        all_channels = torch.arange(qz_masked.size(0), device=qz_masked.device)
        selected_channels = all_channels[all_channels != self.c_anchor]

        random_indices = torch.randperm(len(selected_channels))[:select_channel]
        random_selected_channels = selected_channels[random_indices]

        qz_masked = torch.index_select(qz_masked, 0, random_selected_channels)  # [6, 16, 42]


        qz_anchor_expanded = qz_anchor.unsqueeze(0)  # [1, 16, 42]


        qz_anchor_expanded = qz_anchor_expanded.expand(select_channel, -1, -1)  # [6, 16, 42]

        sim_pos = torch.sum(qz_anchor_expanded * qz_masked, dim=1)  # torch.Size([6, 42])

        # 负样本
        allbs = torch.arange(self.batch_size)
        otherbs = allbs[allbs != self.bs_anchor]


        qz_neg = qz[otherbs, self.c_anchor, :, :]  # [127, 16, 42]


        qz_anchor_expanded = qz_anchor.unsqueeze(0)  # [1, 16, 42]


        qz_anchor_expanded = qz_anchor_expanded.expand(self.batch_size - 1, -1, -1)  # [127, 16, 42]


        sim_neg = torch.sum(qz_anchor_expanded * qz_neg, dim=1)  # torch.Size([127, 42])

        sim_pos_max, _ = torch.max(sim_pos, dim=0, keepdim=True)
        sim_neg_max, _ = torch.max(sim_neg, dim=0, keepdim=True)


        exp_pos = torch.exp((sim_pos - sim_pos_max) / self.temperature)  # [6, 42]
        exp_neg = torch.exp((sim_neg - sim_neg_max) / self.temperature)  # [127, 42]

        sum_exp_pos = torch.sum(exp_pos, dim=0)  # [42]
        sum_exp_neg = torch.sum(exp_neg, dim=0)  # [42]

        # InfoNCE loss
        loss = -torch.log(sum_exp_pos / (sum_exp_pos + sum_exp_neg))  # [42]
        loss = torch.mean(loss)

        return loss


    def forward(self, z):
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)

        x = z


        qz, mu, logvar = self.backbone(z)


        kl_loss = compute_kl_divergence(mu, logvar)


        contrastiveloss = self.contrastive_loss(qz, x, self.select_channel)


        mine_losses = []
        # keypoint = [20, 40]
        for i in range(1, self.patch_num - 1):
            T_real, T_fake = generate_samples(self.mine_model, qz, i, i + 1)
            mine_loss_value = mine_loss(T_real, T_fake)
            mine_losses.append(mine_loss_value)

        cmi_loss = torch.mean(torch.stack(mine_losses))


        cmi_loss = torch.clamp(cmi_loss, max=(self.contraweight * contrastiveloss/2))


        vaeloss = self.klweight * kl_loss + self.contraweight * contrastiveloss - self.cmiweight * cmi_loss




        z = self.head(qz)


        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z, vaeloss

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )

    def increment_epoch(self):

        self.current_epoch += 1


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, flag, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()

        self.flag = flag

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

        self.loc_net = FeedNet(d_model, d_model // 2, type="mlp", n_layers=1)
        self.var_net = nn.Sequential(
            FeedNet(d_model, d_model // 2, type="mlp", n_layers=1),
            nn.Softplus()
        )



        self.mu_avg = None
        self.logvar_avg = None
        self.momentum = 0.9

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        # 数据的Encoder
        if self.flag:
            n_vars = x.shape[1]
            # Input encoding
            x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
            x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

            u = torch.reshape(x, (
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
            u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

            # Encoder
            z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

            realz = z

            bs, nvars, d_model, patchnum = z.size()
            # z = z.reshape(bs * nvars * patchnum, d_model)

            x_reshape = x.reshape(bs*nvars*patchnum, -1)
            mu, logvar = self.loc_net(x_reshape), self.var_net(x_reshape)


            if self.mu_avg is None or self.logvar_avg is None:
                self.mu_avg = mu.mean(dim=0).detach()
                self.logvar_avg = logvar.mean(dim=0).detach()


            mu = torch.where(torch.isnan(mu), self.mu_avg, mu)
            logvar = torch.where(torch.isnan(logvar), self.logvar_avg, logvar)


            valid_mu = mu[~torch.isnan(mu)]
            valid_logvar = logvar[~torch.isnan(logvar)]

            if valid_mu.numel() > 0:
                self.mu_avg = self.momentum * self.mu_avg + (1 - self.momentum) * valid_mu.mean(dim=0).detach()
            if valid_logvar.numel() > 0:
                self.logvar_avg = self.momentum * self.logvar_avg + (1 - self.momentum) * valid_logvar.mean(
                    dim=0).detach()


            self.mu_avg = mu.mean(dim=0).detach()
            self.logvar_avg = logvar.mean(dim=0).detach()

            mu = torch.clamp(mu, min=-2, max=2)
            logvar = torch.clamp(logvar, min=-2, max=2)

            return realz, mu, logvar


        else:
            n_vars = x.shape[1]

            x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
            x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

            u = torch.reshape(x, (
                x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
            u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

            # Encoder
            z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]
            return z



class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v


        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)


        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)


        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))


        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:


        if self.pre_norm:
            src = self.norm_attn(src)

        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)


        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)

        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):

        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)


        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)


        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q


        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)


        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):


    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        attn_scores = torch.matmul(q, k) * self.scale


        if prev is not None: attn_scores = attn_scores + prev


        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask


        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)


        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)


        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

