from nets.CFE_UNet import CFE_UNet
import torch
from thop import profile
import Config as config

if __name__ == "__main__":
    # #call Transception_res

    model = CFE_UNet(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=config.n_filts)
    input = torch.randn(1, 3, 224, 224)
    Flops, params = profile(model, inputs=(input,)) # macs
    print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    print('params参数量: % .4fM'% (params / 1000000))