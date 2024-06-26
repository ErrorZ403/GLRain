from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB



class EDBB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=None, act_type='prelu', with_idt = False, deploy=False, with_13=False, gv=False):
        super(EDBB, self).__init__()
        
        self.deploy = deploy
        self.act_type = act_type
        
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        self.gv = gv          

        if depth_multiplier is None:
            self.depth_multiplier = 1.0
        else: 
            self.depth_multiplier = depth_multiplier   # For mobilenet, it is better to have 2X internal channels
        
        if deploy:
            self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        else: 
            self.with_13 = with_13
            if with_idt and (self.inp_planes == self.out_planes):
                self.with_idt = True
            else:
                self.with_idt = False

            self.rep_conv = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            self.conv1x1 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.deploy:
            y = self.rep_conv(x)
        elif self.gv:
            y = self.rep_conv(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x) + x 
        else:
            y = self.rep_conv(x)     + \
                self.conv1x1(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)            
                #self.conv1x1_3x3(x) + \
            if self.with_idt:
                y += x
            if self.with_13:
                y += self.conv1x1_3x3(x)

        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
    def switch_to_gv(self):
        if self.gv:
            return
        self.gv = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias
        RK, RB = (K0+K5), (B0+B5) 
        if self.with_13:
            RK, RB = RK + K1, RB + B1

        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():
            para.detach_()
      
    
    def switch_to_deploy(self):

        if self.deploy:
            return
        self.deploy = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias
        if self.gv:
            RK, RB = (K0+K2+K3+K4), (B0+B2+B3+B4) 
        else:
            RK, RB = (K0+K2+K3+K4+K5), (B0+B2+B3+B4+B5) 
            if self.with_13:
                RK, RB = RK + K1, RB + B1
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt        
            

        self.rep_conv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():
            para.detach_()
            
        #self.__delattr__('conv3x3')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')

class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

class sESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        c1_ = (self.conv1(x))
        v_max = F.max_pool2d(c1_, kernel_size=7, stride=3)
        c2 = self.conv3(v_max)
        c2 = F.interpolate(c2, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)

        c3 = self.conv4(c2 + c1_)
        m = self.sigmoid(c3)
        return x * m
    
class SpatialAttention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.sa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x * self.sa(x)

        return x
    
class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.c5(out)
        out = self.esa(out)

        return out

class RRFLB(nn.Module):
    def __init__(self,
                in_channels,
                mid_channels=None,
                out_channels=None,
                esa_channels=16,
                attention=None,
                deploy=False):
        super().__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = EDBB(in_channels, mid_channels, deploy=deploy, act_type='linear')
        self.c2_r = EDBB(mid_channels, mid_channels, deploy=deploy, act_type='linear')
        self.c3_r = EDBB(mid_channels, in_channels, deploy=deploy, act_type='linear')

        self.c5 = conv_layer(in_channels, out_channels, 1)
        if attention:
            self.esa = sESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('prelu')

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.c5(out)
        out = self.esa(out)

        return out

class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 num_blocks=4,
                 upscale=4):
        super(RLFN, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)
        rlfn_network = []
        for _ in range(num_blocks):
            rlfn_network.append(RLFB(feature_channels))
        self.rlfn_network = nn.ModuleList(rlfn_network)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)
        
        self.make_output = conv_layer(feature_channels, out_channels, kernel_size=3)#nn.Identity()#conv_layer(feature_channels, out_channels, kernel_size=3)#nn.Identity()

    def forward(self, x):
        out_feature = self.conv_1(x)
        x = out_feature.clone()
        
        for block in self.rlfn_network:
            x = block(x)

        out_low_resolution = self.conv_2(x) + out_feature
        output = self.make_output(out_low_resolution)

        return output
    
class NESR(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 mid_channels=48,
                 num_blocks=4,
                 attention=[],
                 deploy=False):
        super().__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)
        nesr_network = []
        for i in range(num_blocks):
            nesr_network.append(
                RRFLB(feature_channels, mid_channels, deploy=deploy, attention=attention[i])
            )
        self.rlfn_network = nn.ModuleList(nesr_network)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)
        
        self.make_output = conv_layer(feature_channels, out_channels, kernel_size=3)#nn.Identity()#conv_layer(feature_channels, out_channels, kernel_size=3)#nn.Identity()

    def forward(self, x):
        out_feature = self.conv_1(x)
        x = out_feature.clone()
        
        for block in self.rlfn_network:
            x = block(x)

        out_low_resolution = self.conv_2(x) + out_feature
        output = self.make_output(out_low_resolution)

        return output