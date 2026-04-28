import math
import torch
import torch.nn as nn
from torch.nn import init as init
from Network.DilateFormer import Dilateformer
from einops import rearrange, repeat

def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)

    def conv_block(self, input_channels, k):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.LeakyReLU(),
            nn.Conv2d(input_channels, k, kernel_size=3, padding=1))
            
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X

class Dense_path(nn.Module):
    def __init__(self, in_channel=32,k=8):
        super(Dense_path, self).__init__()
        self.Dense = DenseBlock(in_channel=in_channel, k=k) 
        self.final_conv = nn.Conv2d(4*k+ in_channel, 32, 1)

    def forward(self, x):
        x1 = self.Dense(x)     
        x2 = self.final_conv(x1)
        return x2

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, shallow_dim=32):
        super(CAB, self).__init__()

        self.Dense = Dense_path()
        self.cab = ChannelAttention(shallow_dim, shallow_dim//2)

    def forward(self, x):
        return self.cab(self.Dense(x))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class HybridGate(nn.Module):
    def __init__(self, dim=32, mlp_ratio=2):
        super(HybridGate, self).__init__()
        expand_dim = dim * 2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        self.ca = CAB()
        self.expand = nn.Conv2d(in_channels=dim, out_channels=expand_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.expand(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.ca(x1)
        x2 = x2.contiguous().view(b, -1, c)
        x2 = self.mlp(x2)
        x2 = x2.contiguous().view(b, c, h, w)
        out = x1 * x2
        return out

def window_partitions(x, window_size):

    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows

def window_reverses(windows, window_size, H, W):
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]

def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, C, H, W]), torch.zeros([B, C, H, W]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, C, H, W], device=windows.device)

    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res    

class frequency_selection(nn.Module):
    def __init__(self, dim=32, dw=1, norm='backward', act_method=nn.GELU, window_size=None, bias=False):
        super(frequency_selection, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
    
class TFGTBlock(nn.Module):
    def __init__(self):
        super(TFGTBlock, self).__init__()
        
        self.skip_scale1 = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.skip_scale2 = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.MSDF = Dilateformer(H=20, W=1024, Ph=2, Pw=4, in_chans=32, 
                            embed_dim=96, hidden_dim=16,
                            depths=[2], num_heads=[3], kernel_size=3, dilation=[1, 2, 3])
        self.GM = HybridGate()
        self.FM = frequency_selection()
        
    def forward(self, input):
        x1 = self.FM(input) + self.MSDF(input) + self.skip_scale1 * input
        x2 = self.FM(x1) + self.GM(x1)  + self.skip_scale2 * x1
        return x2

class TFGFormer(nn.Module):
    def __init__(self, in_channel=1, shallow_dim=32, num_layers=6):
        super(TFGFormer,self).__init__()
      
        self.shallow_feature = nn.Sequential(nn.Conv2d(in_channel, shallow_dim, 3, 1, 1),
                                        nn.BatchNorm2d(shallow_dim), nn.LeakyReLU())        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TFGTBlock()
            self.layers.append(layer)
        self.out_layer = nn.Sequential(nn.BatchNorm2d(shallow_dim), nn.LeakyReLU(),nn.Conv2d(shallow_dim, in_channel, 3, 1, 1))

    def forward(self, x):
        x1 = self.shallow_feature(x)
        for layer in self.layers:
            skip = x1
            x1 = layer(x1) +skip
        out = self.out_layer(x1)

        mask = torch.zeros_like(x)
        mask[x == 0] = 1
        output = torch.mul(mask, out) + x
           
        return output
