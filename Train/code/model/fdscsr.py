import torch.nn as nn
import torch
from model import common




def make_model(args, parent=False):
    return FDSCSR(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class FDSCB(nn.Module):
    def __init__(self, n_feat=64):
        super(FDSCB, self).__init__()
        # self.oup = n_feat
        self.distilled_channels = int(n_feat // 2)  #32
        self.remaining_channels = int((n_feat - self.distilled_channels) // 2)  #16


        self.c11 = common.default_conv(n_feat, n_feat, 1)
        self.c12 = common.default_conv(n_feat, n_feat, 1)

        self.c31 = common.default_conv(n_feat // 2, n_feat // 2, 3)
        self.c32 = common.default_conv(n_feat // 2, n_feat // 2, 3)
        self.c33 = common.default_conv(n_feat // 4, n_feat // 4, 3)

        self.up = nn.ConvTranspose2d(n_feat//2, n_feat // 4, kernel_size=2, stride=2, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(n_feat  // 4, n_feat // 2, 3,stride=2,padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.dw = nn.Conv2d(n_feat // 4, n_feat // 4, 3, 1, 1, groups=self.remaining_channels, bias=False)
        self.pc = common.default_conv(n_feat // 4, n_feat // 4, 1)
        self.ca = CALayer(channel=n_feat//4,reduction=4)


    def forward(self, x):

        y1 = self.c11(x)
        x1,x2,x3 = torch.split(y1, (self.distilled_channels, self.remaining_channels,self.remaining_channels), dim=1)

        y2 = self.act(self.up(x1))
        y3 = self.down(y2) + x1
        y4 = self.c31(x1) * self.sigmoid(y3)
        y5 = self.c32(y4)

        y6 = self.pc(self.dw(x2))
        y7 = self.c33(self.act(y6))
        y8 = self.ca(y7)

        y9 = torch.cat([y5,y8,x3], dim=1)
        y10 = self.c12(y9)
        return x + y10


class LFFM(nn.Module):
    def __init__(self, n_feat=64):
        super(LFFM, self).__init__()

        self.r1 = FDSCB(n_feat = n_feat)
        self.r2 = FDSCB(n_feat = n_feat)
        self.r3 = FDSCB(n_feat = n_feat)
        self.r4 = FDSCB(n_feat = n_feat)
        self.r5 = FDSCB(n_feat = n_feat)
        self.r6 = FDSCB(n_feat = n_feat)

        self.c13 = nn.Conv2d(6 * n_feat, n_feat, 1, stride=1, padding=0, groups=2)


    def forward(self, x):

        y1 = self.r1(x)
        y2 = self.r2(y1)
        y3 = self.r3(y2)
        y4 = self.r4(y3)
        y7 = self.r5(y4)
        y8 = self.r6(y7)
        #
        cat3 = torch.cat([y1,y2,y3,y4,y7,y8], dim=1)
        output = self.c13(cat3)

        return output + x

class FDSCSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FDSCSR, self).__init__()

        n_colors =3
        kernel_size = 3
        self.scale = args.scale[0]
        n_feat = args.n_feats

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = conv(n_colors, n_feat, kernel_size)

        self.r1 = LFFM(n_feat=n_feat)
        self.r2 = LFFM(n_feat=n_feat)
        self.r3 = LFFM(n_feat=n_feat)
        self.r4 = LFFM(n_feat=n_feat)
        self.r5 = LFFM(n_feat=n_feat)
        self.r6 = LFFM(n_feat=n_feat)


        self.c1 = common.default_conv(6 * n_feat, n_feat, 1)
        self.c2 = common.default_conv(n_feat, n_feat, 3)


        modules_tail = [
            conv(n_feat, self.scale * self.scale * 3, 3),
            nn.PixelShuffle(self.scale),
        ]

        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        y_input1 = self.sub_mean(x)
        y_input = self.head(y_input1)

        y1 = self.r1(y_input)
        y2 = self.r2(y1)
        y3 = self.r3(y2)
        y4 = self.r4(y3)
        y5 = self.r5(y4)
        y6 = self.r6(y5)

        cat = torch.cat([y1,y2,y3,y4,y5,y6], dim=1)

        y7 = self.c1(cat)
        y8 = self.c2(y7)

        output = self.tail(y_input + y8)

        y = self.add_mean(output)

        return y

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))