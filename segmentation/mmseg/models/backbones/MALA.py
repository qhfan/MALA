import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple
from ..builder import BACKBONES
from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

class AddLinearAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()


    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)

        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)
    
class VanillaSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)


        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1) # (b n (h w) (h w))
        res = attn @ v # (b n (h w) d)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.dwconv.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class Block(nn.Module):

    def __init__(self, flag, embed_dim, num_heads, ffn_dim, drop_path=0., layerscale=False, layer_init_value=1e-6):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)
        assert flag in ['l', 'v']
        if flag == 'l':
            self.attn = AddLinearAttention(embed_dim, num_heads)
        else:
            self.attn = VanillaSelfAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1),requires_grad=True)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        '''
        x: B C H W
        '''
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, flags, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging=None,
                 layerscale=False, layer_init_value=1e-6):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.RoPE = RoPE(embed_dim, num_heads)

        # build blocks
        self.blocks = nn.ModuleList([
            Block(flags[i], embed_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()
        sin, cos = self.RoPE((h, w))
        for blk in self.blocks:
            x = blk(x, sin, cos)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            # nn.GELU(),
            # nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            # nn.BatchNorm2d(embed_dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)#(b c h w)
        return x

@BACKBONES.register_module()     
class RankBoostLinearTransformerAttn(nn.Module):

    def __init__(self, in_chans=3, out_indices=(0, 1, 2, 3), flagss=[['l']*10, ['l']*10, ['v', 'v']*20, ['v']*10],
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, 
                 projection=1024, layerscales=[False, False, False, False], layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6], norm_eval=True):
        super().__init__()

        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                flags=flagss[i_layer],
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_value=layer_init_values[i_layer]
            )
            self.layers.append(layer)
            
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(embed_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)
            if i in self.out_indices:
                x_out = self.extra_norms[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    
