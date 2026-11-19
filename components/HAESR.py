import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import  torch
import  torch.nn as nn
from    ops.ARB import ARB
from    ops.IRB import IRB
from    ops.pixelshuffle import pixelshuffle_block
import  torch.nn.functional as F

        
class HAESR(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,**kwargs):
        super(HAESR, self).__init__()

        res_num     = kwargs["res_num"]
        up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]

        self.res_num    = res_num
        self.block0 = IRB(channel_num=num_feat,**kwargs)

        self.block1 = ARB(channel_num=num_feat,**kwargs)
        self.block2 = ARB(channel_num=num_feat,**kwargs)

        self.input  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        self.window_size   = kwargs["window_size"]
        self.up_scale = up_scale
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual= self.input(x)

        out, a1, a2, a3, a4, a5, a6 = self.block0(residual)
        out = self.block1(out, a1, a2, a3, a4, a5, a6)
        out = self.block2(out, a1, a2, a3, a4, a5, a6)

        out     = torch.add(self.output(out),residual)
        out     = self.up(out)
        
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        return  out


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from thop import profile
    from thop import clever_format
    upscale = 2
    height = 1280 // upscale
    width = 720 // upscale

    x = torch.rand(1, 3, 720//upscale, 1280//upscale).cuda()

    net = HAESR(num_feat=48, upsampling=upscale, window_size=8, res_num=3, block_num=1, bias=True).cuda()

    net.eval()
    print(net)

    flops = FlopCountAnalysis(net, x)

    fflops = flops.set_op_handle("aten::layer_norm", lambda inputs, outputs: 0)

    print("FLOPs: %.2f" %(flops.total() / 1e9), ' G')
    total = sum([param.nelement() for param in net.parameters()])
    print('Number of params: %.2fK' % (total / 1e3))

