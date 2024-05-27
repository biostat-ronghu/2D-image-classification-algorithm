import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional



def drop_path_f(x,drop_prob: float = 0., training: bool = False):    # drop_path是将一个batch中的一部分样本丢弃
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # ndim返回的是张量是几维张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # rand是[0,1]分布的随机数,randn是N(0,1)分布的随机数
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # random_tensor 的元素取值范围是(0,2),floor是向下取整
    random_tensor.floor_()  # binarize   random_tensor 的元素变成了0和1,取1的概率是keep_prob
    # 广播机制, random_tensor: [x.shape[0],1,...,1] -> [x.shape[0],x.shape[1],...,x.shape[x.ndim - 1]]
    # random_tensor广播之前维度为[x.shape[0],1,...,1],且元素为0(图片丢弃)或1(图片保留),1的比例大概占keep_prob
    # x除以keep_prob,相当于提高保留图片的权重
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
        """
    def __init__(self,drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob=drop_prob

    def forward(self,x):
        return drop_path_f(x, self.drop_prob, self.training)



def window_partition(x,window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows



def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一堆window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, in_c=3, embed_dim=96,norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)     # 方阵图片
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_c,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):         # x: (batch_size,in_chans,image_h,image_w)
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # pad是从后往前，从左往右，从上往下，原顺序是（B,C,H,W) pad顺序就是(W，H，C）
            x = F.pad(x,( 0, self.patch_size[1] - W % self.patch_size[1],             # W右侧填充
                          0, self.patch_size[0] - H % self.patch_size[0],             # W下侧填充
                          0, 0))             # C无填充

        # 下采样patch_size倍
        x = self.proj(x)      # (batch_size,in_chans,image_h,image_w) -> (batch_size,embed_dim,image_h/patch_size,image_w/patch_size)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1,2)       # -> (batch_size,(image_h/patch_size)*(image_w/patch_size),embed_dim)
        x = self.norm(x)
        return x, H, W        # 这里是经过padding的H和W



class PatchMerging(nn.Module):          # PatchMerging运算是将图片的H和W都做两倍下采样,得到4张H和W相同的子图,然后对4张子图在通道维度上拼接(通道数翻4倍)并做layernorm,最后通道数减半
    r""" Patch Merging Layer.
        Args:
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.norm = norm_layer(4*dim)
        self.reduction = nn.Linear(4*dim,2*dim,bias=False)   # 将通道数由4倍变为2倍

    def forward(self,x,H,W):
        """
        x: B, H*W (L) , C, 并不知道H和W，所以需要单独传参
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B,H,W,C)       #  -> (B,H,W,C)

        # padding
        # 因为是下采样两倍，如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 此时(B,H,W,C)依然是从后向前
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))      # 对W右侧和H下侧填充

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]    原图的奇数行奇数列
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]    原图的偶数行奇数列
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]    原图的奇数行偶数列
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]    原图的偶数行偶数列
        # x = torch.cat([x0, x1, x2, x3], -1)
        x = torch.concat([x0,x1,x2,x3],axis=-1)   # [B, H/2, W/2, 4*C],这里的-1就是在C的维度上拼接
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)  # -> [B, H/2*W/2, 4*C]
        x = self.reduction(x)  # -> [B, H/2*W/2, 2*C]

        return x



class Mlp(nn.Module):       # Swin Transformer Block中的MLP部分
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
        """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features     # out_features or in_features: out_features不为None时,等于out_features;否则等于in_features
        hidden_features = hidden_features or in_features         # hidden_features or in_features: hidden_features不为None时,等于hidden_features;否则等于in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.
        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
    def __init__(self,dim, window_size, num_heads, qkv_bias=True,attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads   # embed_dim_per_head 每个头的特征维度
        self.scale = head_dim ** (-0.5)

        # 每一个head都有自己的relative_position_bias_table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [(2*Mh-1) * (2*Mw-1), nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # meshgrid生成网格，再通过stack方法拼接
        # meshgrid([x,y],indexing="ij"),x和y为一维张量, 返回一个长度为2的元组, 每个元组的行数与x的长度相同且列数与y的长度相同
        # stack对张量序列进行拼接
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # -> [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # -> [2, Mh*Mw]  得到绝对位置索引: tensor([[0,...,0,1,...,1,...,6,...,6],\n[0,...,6,0,...,6,...,0,...,6]])
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw] 广播机制     "各位置(比如[0][0])pixel的索引减本窗口所有位置pixel的索引"
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # -> [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # -> [Mh*Mw, Mh*Mw, 2]    得到相对位置索引矩阵 [窗口行索引、窗口行索引、二维相对位置索引]
        # 二维相对位置索引转化为一维相对位置索引
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0    相对位置行索引+窗口高-1
        relative_coords[:, :, 1] += self.window_size[1] - 1           # 相对位置列索引+窗口宽-1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1       # 相对位置行索引 * (2*窗口高-1)
        relative_position_index = relative_coords.sum(-1)  # -> [Mh*Mw, Mh*Mw]  在最后一个维度求和
        # 整个训练当中，window_size大小不变，因此这个索引也不会改变, 将 relative_position_index 放入缓存
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # VIT源码中W_Q、W_K、W_V是分开的,此处self.qkv相当于W_Q、W_K、W_V合并在一起
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多头融合
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)   # 用截断正态分布初始化relative_position_bias_table
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim] ->
        B_, N, C = x.shape     # B_=batch_size*num_windows, N=Mh*Mw, C=total_embed_dim
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]    对应[QKV维度,新batch_size,head个数,token个数,token的向量表示的维度]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # 通过unbind分别获得qkv,把qkv按第一个维度拆分为q、k、v
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)  3个 -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]

        # self.relative_position_index.view(-1)          [Mh*Mw, Mh*Mw] -> Mh*Mw*Mh*Mw 一维张量
        # self.relative_position_bias_table[self.relative_position_index.view(-1)]   [(2*Mh-1) * (2*Mw-1), nH] -> [Mh*Mw*Mh*Mw, nH]
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # -> [nH, Mh*Mw, Mh*Mw]
        # 通过unsqueeze加上一个batch维度, 对于同一批图片(batch),相对位置偏置只随head的不同而不同
        attn = attn + relative_position_bias.unsqueeze(0)     # -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]

        # 有mask表示是SW-MSA
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            # mask.unsqueeze(1).unsqueeze(0) 直接升维,是因为掩码反映了窗口中的任意两个pixel是否来自图片的同一个区域,而与pixel属于哪个head和图片是batch中的哪一个无关
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)    # 将属于不同window的元素之间的注意力值-100,之后softmax得到近似为0
            attn = attn.view(-1, self.num_heads, N, N)     # -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]   得到当前新batch(原batch下所有window,每个window内做注意力)各head所对应的各token与其他token的"相关性"
            attn = self.softmax(attn)           # 对最后一个维度softmax  相当于同一token与其他token的相关程度做归一化
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)     # -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = self.proj_drop(x)
        return x



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,dim,num_heads,window_size=7,shift_size=0,mlp_ratio=4.,qkv_bias=True,
                 drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self,x,attn_mask):
        # x(B,L,C)，因此需要记录h和w
        H, W = self.H, self.W     # self.H和self.W是在BasicLayer的forward中定义的
        B,L,C = x.shape
        assert L == H * W, "input feature has wrong size"
        # 残差
        shortcut = x
        x = self.norm1(x)
        x = x.view(B,H,W,C)    # -> (B,H,W,C)

        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x,(0,0,pad_l,pad_r,pad_t,pad_b))       # -> (B,Hp,Wp,C)
        _, Hp, Wp, _ = x.shape

        # cyclic shift, 不改变维度
        if self.shift_size>0:
            # 对窗口进行移位。窗口从上向下移，从左往右移，因此是负的
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))    # roll函数要求shifts和dims的维度相同,将x的元素在dims维度上从shifts的初始索引开始排列
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x,self.window_size)   #  (B,Hp,Wp,C) -> [B*num_windows,Mh,Mw,C]
        x_windows = x_windows.view(-1,self.window_size * self.window_size,C)    # -> [B*num_windows,Mh*Mw,C]   看成[batch_size,token个数,单个token对应的特征维度]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # -> [nW*B, Mh*Mw, C]

        # 窗口还原
        attn_windows = attn_windows.view(-1,self.window_size,self.window_size,C)    # -> [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)    # -> [B, Hp, Wp, C]

        # shift还原，如果没有shifted就不用还原
        if self.shift_size>0:
            x = torch.roll(shifted_x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        else:
            x = shifted_x

        if pad_r>0 or pad_b>0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicLayer(nn.Module):      # 单个stage的实现
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of Swin Transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): 是否需要下采样，在最后一个stage不需要. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size//2     # 移动尺寸,移动尺寸的计算公式是论文规定的,等于窗口除以2向下取整

        # 在当前stage之中所有的Swin Transformer block
        # 注意每个Swin Transformer block中只会有一个MSA,要么W-MSA，要么SW-MSA，所以shift_size为0代表W-MSA，不为0代表SW-MSA

        self.blocks = nn.ModuleList([ SwinTransformerBlock(dim=dim,
                                                           num_heads=num_heads,
                                                           window_size=window_size,
                                                           shift_size=0 if (i%2==0) else self.shift_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           drop=drop,
                                                           attn_drop=attn_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path,
                                                           norm_layer=norm_layer)  for i in range(depth) ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def create_mask(self,x,H,W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        # 将图片划分成9个区域,各区域中的值分别赋值为0-8,作为各pixel的区域标识码
        for h in h_slices:        # h,w都是切片
            for w in w_slices:
                img_mask[:,h,w,:]+=cnt
                cnt+=1

        mask_windows = window_partition(img_mask,self.window_size)  # 窗口划分 -> [number_Window, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1,self.window_size*self.window_size)  # [number_Window, Mh*Mw] 窗口展平
        # 相减是在判断各window中的任意两个pixel在计算Attention时是不是来源于同一个区域(一个Mh*Mw的window共有Mh*Mw个pixel),所以维度是[number_Window,Mh*Mw,Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [number_Window,1,Mh*Mw] - [number_Window,Mh*Mw,1]
        # 相减的两个张量维度不同,广播机制 [number_Window,Mh*Mw,Mh*Mw]
        # masked_fill(cond,value)  元素满足cond时,用value填充
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self,x,H,W):
        # 先创建一个mask蒙版，在图像尺寸不变的情况下蒙版也不改变
        attn_mask = self.create_mask(x, H, W)  # -> [number_Window, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # 默认不使用checkpoint方法
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            # downsample其实是patch merging, 图片的H和W减半且通道数翻倍
            x = self.downsample(x, H, W)      #  batch_size维度不变,H和W减半,C翻倍
            # 防止H和W是奇数。如果是奇数，在下采样中经过一次padding就变成偶数了，但如果这里不给H和W加一的话就会导致少一个，如果是偶数，加一除二取整还是不变
            H, W = (H + 1) // 2, (W + 1) // 2    # 取整除

        return x, H, W



class SwinTransformer(nn.Module):
    r""" Swin Transformer
        Args:
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Swin Transformer可以作为一个通用骨架，在这里将其用在分类任务中，最后分为num_classes个类. Default: 1000
            embed_dim (int): Patch embedding dimension，就是原文中的C. Default: 96
            depths (tuple(int)): 每个stage中的Swin Transformer Block数.
            num_heads (tuple(int)): 每个stage中用的multi-head数.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): mlp的隐藏层是输入层的多少倍. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): 在pos_drop,mlp及其他地方. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): 每一个Swin Transformer Block之中，注意它的dropout率是递增的. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): 如果使用可以节省内存. Default: False
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,      # embed_dim等于 patch_size*patch_size*3*2
                 depths=(2,2,6,2),
                 num_heads=(3,6,12,24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 **kwargs
                 ):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features =  int(embed_dim*2**(self.num_layers-1))
        self.mlp_ratio = mlp_ratio
        # 对应Patch partition和Linear Embedding
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 在每个Swin Transformer Block的dropout率，是一个递增序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]      # 把[0,drop_path_rate]均分，得到长度为sum(depths)的等差数列(数列的最小和最大分别是0和drop_path_rate)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):    # i_layer是stage索引,从0开始
            # num_layers及stage数
            # 与论文不同，代码中的stage包含的是下一层的Patch merging ，因此在最后一个stage中没有Patch merging
            # dim为当前stage的维度，depth是当前stage堆叠多少个block，drop_patch是本层所有block的drop_patch
            # downsample是Patch merging，并且在最后一个stage为None
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],                # depth 当前stage的Swin Transformer block个数
                                num_heads=num_heads[i_layer],         # num_heads 当前stage的head个数
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[ sum(depths[:i_layer]) : sum(depths[:i_layer + 1]) ],       # drop_path 当前stage的各Swin Transformer block的 drop_out概率序列
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,     # 最后一层不做PatchMerging
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        # 针对分类任务
        self.norm = norm_layer(self.num_features)
        # 在这个分类任务中，用全局平均池化取代cls token
        self.avgpool = nn.AdaptiveAvgPool1d(1)      # nn.AdaptiveAvgPool1d(L_out) 对输入进行1维自适应平均池化，输入维度(N,C,L_in),输出维度(N,C,L_out)
        self.head =  nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)     # 应用自定义的权重初始化方案


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):     # 如果m属于Linear类的实例
            nn.init.trunc_normal_(m.weight, std=.02)      # 权重按截取正态分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)    # 偏置赋值为常数0
        elif isinstance(m, nn.LayerNorm):      # 如果m属于LayerNorm类的实例
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self,x):         # x: (batch_size,in_chans,image_h,image_w)
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # 依次通过每个stage
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)         # -> (batch_size, image_h*image_w/( 2**(stage个数-1) * 2**(stage个数-1) ) ,num_features)
        x = self.avgpool(x.transpose(1,2))    # -> (batch_size,num_features,1)        这里没有取class_token,而是对所有"token"做了平均池化
        x = torch.flatten(x,1)     # -> (batch_size,num_features)
        x = self.head(x)     # (batch_size,num_features) -> (batch_size,num_classes)
        return x



# def main():
#     swin_tiny_patch4_window7_224 = SwinTransformer(patch_size=4,in_chans=3,num_classes=20,embed_dim=96,
#                                                    depths=(2,2,6,2),num_heads=(3,6,12,24),window_size=7,
#                                                    use_checkpoint=False)
#     img_input = torch.randn((10,3,224,224))
#     out_put = swin_tiny_patch4_window7_224(img_input)
#     print(out_put.shape)
#     print(out_put)
#     out_put = out_put.argmax(dim=1)
#     print(out_put)
#
#
#
# if __name__ == '__main__':
#     main()