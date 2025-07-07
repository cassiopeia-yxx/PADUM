# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2024/6/14 10:34
# File : dutn.py



import math
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import numbers
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from basicsr.models.archs.mamba import *

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)
        # self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class FFT_FFS(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(FFT_FFS, self).__init__()
        self.x_amp_fuse = nn.Sequential(
            nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups= in_nc),
        )
        self.x_pha_fuse = nn.Sequential(
            nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups= in_nc),
        )
        self.gamma = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups= in_nc)
        self.phi = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups= in_nc)

    def forward(self, x, x_enc, x_dec):

        _, _, H, W = x.shape

        # 计算频域幅值和相位
        x_enc_fft = torch.fft.rfft2(x_enc, norm='backward')
        x_dec_fft = torch.fft.rfft2(x_dec, norm='backward')
        x_freq_amp = torch.abs(x_enc_fft)
        x_freq_pha = torch.angle(x_dec_fft)

        # 融合频域幅值和相位
        x_freq_amp = self.x_amp_fuse(x_freq_amp)
        x_freq_pha = self.x_pha_fuse(x_freq_pha)


        # 计算实部和虚部
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)

        # 创建复数张量并进行逆变换
        x_recom = torch.complex(real, imag)
        x_recom = torch.fft.irfft2(x_recom, s=(H, W), norm='backward')

        out = self.gamma(x_recom) * x + self.phi(x_recom) + x
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2



class Bi_Mamba(nn.Module):
    def __init__(self,
            dim,
            ssm_cfg,
            depth,
            norm_epsilon,
            rms_norm,
            residual_in_fp32,
            fused_add_norm,
            bimamba_type,
            drop_path,
            norm_layer,
            device,
            dtype,
            use_checkpoint,
            img_size,
            patch_size,
            resi_connection):
        super(Bi_Mamba, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=norm_layer)

        self.mamba = RMB(
            dim=dim,
            ssm_cfg=ssm_cfg,
            depth=depth,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            bimamba_type=bimamba_type,
            drop_path=drop_path,
            norm_layer=norm_layer,
            device=device,
            dtype=dtype,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=nn.LayerNorm)

        self.norm1 = LayerNorm(dim)
        self.drop_path1 = nn.Dropout(drop_path)
        self.drop_path2 = nn.Dropout(drop_path)
        self.norm2 = LayerNorm(dim)
        self.dwcon = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
        )
        self.ffn = FeedForward(dim, ffn_expansion_factor=2.66, bias=False)
        # ffn_channel = 2 * dim
        # self.ffn = nn.Sequential(
        #     nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        #     SimpleGate(),
        #     nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # )



    def forward(self, input):
        # first
        x_size = (input.shape[2], input.shape[3])
        x_norm = self.norm1(input)

        # mamba
        x_mamba = self.patch_embed(x_norm)
        x_mamba = self.mamba(x_mamba, x_size)
        x_mamba = self.patch_unembed(x_mamba, x_size)

        x = input + self.drop_path1(self.dwcon(x_norm) * x_mamba)
        # final
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class UNetConvBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 downsample,
                 use_csff=False,
                 img_size=128,
                 depths=[1,1,1,1],
                 i_layer=1,
                 patch_size=1,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 residual_in_fp32=False,
                 fused_add_norm=False,
                 bimamba_type="none", # v2
                 norm_layer=nn.LayerNorm,
                 device=None,
                 dtype=None,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.RSSB = Bi_Mamba(
            dim=in_size,
            ssm_cfg=ssm_cfg,
            depth=depths[i_layer],
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            bimamba_type=bimamba_type,
            drop_path=dpr[i_layer],
            norm_layer=norm_layer,
            device=device,
            dtype=dtype,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            )

        if downsample and use_csff:
            self.FFT_INT = FFT_FFS(in_size, in_size)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)


    def forward(self, x, enc=None, dec=None):
        out = x
        out = self.RSSB(out)  # run mamba
        if enc is not None and dec is not None:
            assert self.use_csff
            out = self.FFT_INT(out, enc, dec)  # run FFT_FFS
        if self.downsample:
            out_down = self.downsample(out)  # run Down
            return out_down
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, depth, img_size, depths, i_layer):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv = nn.Conv2d(out_size*2, out_size, 1, bias=False)
        self.conv_block = UNetConvBlock(out_size, out_size, downsample=False, use_csff=False, img_size=img_size, depths=depths, i_layer=i_layer)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv(torch.cat([up, bridge], dim=1))
        out = self.conv_block(out)
        return out


class Encoder(nn.Module):
    def __init__(self, n_feat, use_csff=False, depth=3, img_size=64):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        depths = [1 for _ in range(depth)]
        for i in range(depth - 1):
            self.body.append(UNetConvBlock(in_size=n_feat * 2 ** (i), out_size=n_feat * 2 ** (i + 1), downsample=True,
                                           use_csff=use_csff, img_size=img_size, depths=depths, i_layer=i))

        self.body.append(
            UNetConvBlock(in_size=n_feat * 2 ** (depth - 1), out_size=n_feat * 2 ** (depth - 1), downsample=False,
                          use_csff=use_csff, img_size=img_size, depths=depths, i_layer=depth - 1))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x, encoder_outs[i], decoder_outs[-i - 1])
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x)
                else:
                    x = down(x)
        return res, x


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, depth=3, img_size=64):
        super(Decoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        depths = [1 for _ in range(depth)]
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat * 2 ** (depth - i - 1), out_size=n_feat * 2 ** (depth - i - 2),
                                         depth=depth - i - 1, img_size=img_size, depths=depths, i_layer=i))
    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, bridges[-i - 1])
            res.append(x)
        return res, x



class Bnet(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, n_depth, img_size=64):
        super(Bnet, self).__init__()
        self.shallow_encoder1 = nn.Sequential(conv(in_c, out_c, kernel_size, bias=False),
                                              CAB(out_c, kernel_size, reduction=4, bias=False, act=nn.PReLU()))
        self.stage_encoder = Encoder(out_c, use_csff=True, depth=n_depth, img_size=img_size)
        self.stage_decoder = Decoder(out_c, kernel_size, depth=n_depth, img_size=img_size)
        self.shallow_decoder1 = conv(out_c, 3, kernel_size, bias=False)
    def forward(self, B, f_encoder, f_decoder):
        B = self.shallow_encoder1(B)
        feat1, f_encoder = self.stage_encoder(B, f_encoder, f_decoder)
        f_decoder, last_out = self.stage_decoder(f_encoder, feat1)
        last_out = self.shallow_decoder1(last_out)
        return last_out, feat1, f_decoder


class Rnet(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, n_depth, img_size=64):
        super(Rnet, self).__init__()
        self.stage_encoder = Encoder(out_c, use_csff=True, depth=n_depth, img_size= img_size)
        self.stage_decoder = Decoder(out_c, kernel_size, depth=n_depth, img_size= img_size)

    def forward(self, R, f_encoder, f_decoder):
        feat1, f_encoder = self.stage_encoder(R, f_encoder, f_decoder)
        f_decoder, last_out = self.stage_decoder(f_encoder, feat1)
        return last_out, feat1, f_decoder

class init_B(nn.Module):
    def __init__(self, in_c, n_feat, kernel_size, n_depth, img_size):
        super(init_B, self).__init__()
        self.shallow_encoder = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=False),
                                             CAB(n_feat, kernel_size, reduction=4, bias=False, act=nn.PReLU()))
        self.init_encoder = Encoder(n_feat, use_csff=False, depth=n_depth, img_size= img_size)
        self.init_decoder = Decoder(n_feat, kernel_size, depth=n_depth, img_size= img_size)
        self.shallow_decoder = conv(n_feat, 3, kernel_size, bias=False)
    def forward(self, B):
        B = self.shallow_encoder(B)
        feat1, f_encoder = self.init_encoder(B)
        f_decoder, out_put = self.init_decoder(f_encoder, feat1)
        out_put = self.shallow_decoder(out_put)
        return feat1, f_decoder, out_put

class init_R(nn.Module):
    def __init__(self, n_feat, kernel_size, n_depth, img_size=64):
        super(init_R, self).__init__()

        self.init_encoder = Encoder(n_feat, use_csff=False, depth=n_depth, img_size= img_size)
        self.init_decoder = Decoder(n_feat, kernel_size, depth=n_depth, img_size= img_size)

    def forward(self, R):
        feat1, f_encoder = self.init_encoder(R)
        f_decoder, out_put = self.init_decoder(f_encoder, feat1)
        return feat1, f_decoder, out_put


class feature_ext(nn.Module):
    def __init__(self, n_feat, output):
        super(feature_ext, self).__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, 1),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat),
            nn.GELU(),
            conv(n_feat, n_feat, 1),

            conv(n_feat, n_feat, 1),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat),
            nn.GELU(),
            conv(n_feat, n_feat, 1),
        )
        self.conv1 = conv(n_feat, output, 1)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.body(x) + x
        x = self.conv1(x)
        out = self.act1(x)
        return out

class PADM(nn.Module):
    def __init__(self,
                 img_size =256,
                 in_c = 3,
                 n_feat = 32,
                 nums_stage = 4,
                 n_depth =3,
                 kernel_size=3):
        super(PADM, self).__init__()
        self.nums_stages = nums_stage

        self.init_updateB = init_B(in_c, n_feat, kernel_size, n_depth, img_size)
        self.init_etaB = feature_ext(3, 3)
        self.init_etaR = feature_ext(3, n_feat)
        self.init_updateR = init_R(n_feat, kernel_size, n_depth, img_size)
        self.get_z0 = nn.Sequential(conv(in_c, n_feat, kernel_size),
                                    nn.PReLU())
        self.dt0 = conv(in_c, n_feat, kernel_size)
        self.d0 = conv(n_feat, in_c, kernel_size)
        self.get_R0 = conv(n_feat, in_c, kernel_size)

        # Stage 2 to k-1
        self.proxNet_B = nn.ModuleList([Bnet(in_c=in_c, out_c=n_feat, kernel_size=kernel_size, n_depth=n_depth, img_size=img_size) for _ in range(self.nums_stages)])
        self.proxNet_R = nn.ModuleList([Rnet(in_c=in_c, out_c=n_feat, kernel_size=kernel_size, n_depth=n_depth, img_size=img_size) for _ in range(self.nums_stages)])
        self.dt_S = nn.ModuleList([conv(in_c, n_feat, kernel_size) for _ in range(self.nums_stages)])
        self.d_S = nn.ModuleList([conv(n_feat, in_c, kernel_size) for _ in range(self.nums_stages)])
        self.get_RS = nn.ModuleList([conv(n_feat, in_c, kernel_size) for _ in range(self.nums_stages)])
        self.etaB_S = nn.ModuleList([feature_ext(n_feat, 3) for _ in range(self.nums_stages)])
        self.etaR_S = nn.ModuleList([feature_ext(n_feat, n_feat) for _ in range(self.nums_stages)])


    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f)
        return eta


    def forward(self, I):
        output_B = []
        output_R = []
        b, c, h, w = I.shape
        hb, wb = 32, 32
        pad_h = (hb - h % hb) % hb
        pad_w = (wb - w % wb) % wb
        I = F.pad(I, [0, pad_w, 0, pad_h], mode='reflect')

        B = I
        R = torch.zeros_like(I)

        ##----------- stage 1 ----------------- ##
        # update B
        B_hat = (torch.ones_like(I) - self.init_etaB(I)) * B + self.init_etaB(I) * (I - R)
        feat_B, f_decoder_B, out_put_B = self.init_updateB(B_hat)
        B = B_hat + out_put_B

        # update R
        R0 = I - B
        z0 = self.get_z0(R0)
        z_hat = z0 - self.init_etaR(R0) * self.dt0((self.d0(z0) - R0))
        feat_R, f_decoder_R, out_put_z = self.init_updateR(z_hat)
        z = out_put_z + z_hat
        R = self.get_R0(z)

        output_B.append(B[:, :, :h, :w])
        output_R.append(R[:, :, :h, :w])
        ##-------------- Stage 2 to k-1---------------------##
        for i in range(self.nums_stages):

            # update B
            B_hat = (torch.ones_like(I) - self.etaB_S[i](f_decoder_B[-1])) * B + self.etaB_S[i](f_decoder_B[-1]) * (I - R)
            out_put_B, feat_B, f_decoder_B = self.proxNet_B[i](B_hat, feat_B, f_decoder_B)
            B = B_hat + out_put_B

            # update R
            z_hat = z - self.etaR_S[i](f_decoder_R[-1]) * self.dt_S[i]((self.d_S[i](z) - (I - B)))
            out_put_z, feat_R, f_decoder_R = self.proxNet_R[i](z_hat, feat_R, f_decoder_R)
            z = out_put_z + z_hat
            R = self.get_RS[i](z)
            output_B.append(B[:, :, :h, :w])
            output_R.append(R[:, :, :h, :w])
        return output_B, output_R
