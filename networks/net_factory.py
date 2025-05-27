try:
    from unet import UNet, UNet_DTPC
except:
    from .unet import (UNet, UNet_DTPC)

def net_factory(net_type="unet", in_chns=3, class_num=2):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_DTPC":
        net = UNet_DTPC(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
