import torch.nn as nn
from ops.esa import ESA
from ops.MPAB import AttentionReuseStages



class ARB(nn.Module):
    def __init__(self, channel_num=64, bias = True, block_num=4,**kwargs):
        super(ARB, self).__init__()

        ffn_bias    = kwargs.get("ffn_bias", False)
        window_size = kwargs.get("window_size", 0)
        pe          = kwargs.get("pe", False)

        self.residual_layer = AttentionReuseStages(channel_num, bias=bias, ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
        self.res_end = nn.Conv2d(channel_num,channel_num,1,1,0,bias=bias)
        esa_channel     = max(channel_num // 4, 16)
        self.esa        = ESA(esa_channel, channel_num)
        
    def forward(self, x, a1, a2, a3, a4, a5, a6):
        out = self.residual_layer(x, a1, a2, a3, a4, a5, a6)
        out = self.res_end(out)
        out = out + x
        return self.esa(out)