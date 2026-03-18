"""Transformer backbone for DDI-E - adapted from Diffusion-TS"""
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import (
    LearnablePositionalEncoding, Conv_MLP, AdaLayerNorm, Transpose, GELU2, series_decomp
)


class TrendBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super().__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_dim, trend_poly, 3, padding=1), act,
            Transpose(shape=(1, 2)), nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )
        lin_space = torch.arange(1, out_dim + 1, 1, dtype=torch.float) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** (p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        return trend_vals.transpose(1, 2)


class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model, self.factor, self.low_freq = d_model, factor, low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq, f = x_freq[:, self.low_freq:-1], torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq, f = x_freq[:, self.low_freq:], torch.fft.rfftfreq(t)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = max(1, int(self.factor * math.log(length)))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_arr = rearrange(torch.arange(t, dtype=torch.float, device=x_freq.device), 't -> () () t ()')
        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t_arr + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')


class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), att.mean(dim=1, keepdim=False)


class CrossAttention(nn.Module):
    def __init__(self, n_embd, condition_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), att.mean(dim=1, keepdim=False)


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, activate='GELU'):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd), act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd), nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, _ = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x)), None


class Encoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate):
        super().__init__()
        self.blocks = nn.Sequential(*[
            EncoderBlock(n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
            for _ in range(n_layer)
        ])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block in self.blocks:
            x, _ = block(x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_channel, n_feat, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, activate, condition_dim):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln1_1 = AdaLayerNorm(n_embd)
        self.attn1 = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn2 = CrossAttention(n_embd, condition_dim, n_head, attn_pdrop, resid_pdrop)
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd), act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd), nn.Dropout(resid_pdrop),
        )
        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, _ = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, _ = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    def __init__(self, n_channel, n_feat, n_embd, n_head, n_layer, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, condition_dim):
        super().__init__()
        self.d_model, self.n_feat = n_embd, n_feat
        self.blocks = nn.Sequential(*[
            DecoderBlock(n_channel, n_feat, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, condition_dim)
            for _ in range(n_layer)
        ])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        mean, season, trend = [], torch.zeros((b, c, self.d_model), device=x.device), torch.zeros((b, c, self.n_feat), device=x.device)
        for block in self.blocks:
            x, residual_mean, residual_trend, residual_season = block(x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)
        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(self, n_feat, n_channel, n_layer_enc=5, n_layer_dec=14, n_embd=1024, n_heads=16,
                 attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, block_activate='GELU', max_len=2048, conv_params=None, **kwargs):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        kernel_size = conv_params[0] if conv_params else 1
        padding = conv_params[1] if conv_params else 0
        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size, stride=1, padding=padding, padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, 1, stride=1, padding=0, padding_mode='circular', bias=False)
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, input, t, padding_masks=None, return_res=False):
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend
        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m
        return trend, season_error
