import torch
import torch.nn as nn


class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out



class UpsamplingAndDownsampling(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, k):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in Upsampling and Downsampling
        """

        super(UpsamplingAndDownsampling , self).__init__()

        self.k = k

        self.cnv = torch.nn.Conv2d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=2,
            )

        elif self.k == 3:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                ],
                dim=2,
            )

        elif self.k == 4:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                ],
                dim=2,
            )

        elif self.k == 5:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.AvgPool2d(16)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.MaxPool2d(16)(x)),
                ],
                dim=2,
            )

        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x



class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)



    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.activation(x)


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers        
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        #self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.activation(x)


class ChannelShuffle(nn.Module):
    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.num_groups

        # Reshape
        x = x.view(batch_size, self.num_groups, channels_per_group, height, width)

        # Transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # Flatten
        x = x.view(batch_size, -1, height, width)

        return x



class MLFC(torch.nn.Module):

    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'LeakyReLU'.
            k (int, optional): k in Upsampling and Downsampling. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in Upsampling and Downsampling. Defaults to 4.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)
        self.channel_shuffle = ChannelShuffle(n_filts * inv_fctr)

        self.uad = UpsamplingAndDownsampling(n_filts * inv_fctr, n_filts, k)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU()


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.channel_shuffle(x)
        x = self.uad(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls))


    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))  #这里有问题，两次归一化

class CFE_UNet(torch.nn.Module):

    def __init__(self, n_channels, n_classes, n_filts=32):
        """
        Initialization

        Args:
            n_channels (int): number of channels of the input image.
            n_classes (int): number of output classes
            n_filts (int, optional): multiplier of the number of filters throughout the model.
                                     Increase this to make the model wider.
                                     Decrease this to make the model ligher.
                                     Defaults to 32.
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = MLFC(n_channels, n_filts, k=3, inv_fctr=3)
        self.cnv12 = MLFC(n_filts, n_filts, k=3, inv_fctr=3)

        self.cnv21 = MLFC(n_filts, n_filts * 2, k=3, inv_fctr=3)
        self.cnv22 = MLFC(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)
        self.ca1 = ChannelAttention(n_filts * 2)
        self.sa = SpatialAttention()

        self.cnv31 = MLFC(n_filts * 2, n_filts * 4, k=3, inv_fctr=3)
        self.cnv32 = MLFC(n_filts * 4, n_filts * 4, k=3, inv_fctr=3)
        self.ca2 = ChannelAttention(n_filts * 4)

        self.cnv41 = MLFC(n_filts * 4, n_filts * 8, k=2, inv_fctr=3)
        self.cnv42 = MLFC(n_filts * 8, n_filts * 8, k=2, inv_fctr=3)
        self.ca3 = ChannelAttention(n_filts * 8)

        self.cnv51 = MLFC(n_filts * 8, n_filts * 16, k=1, inv_fctr=3)
        self.cnv52 = MLFC(n_filts * 16, n_filts * 16, k=1, inv_fctr=3)
        self.ca4 = ChannelAttention(n_filts * 16)

        self.rspth1 = ResPath(n_filts, 1)
        self.rspth2 = ResPath(n_filts * 2, 1)
        self.rspth3 = ResPath(n_filts * 4, 1)
        self.rspth4 = ResPath(n_filts * 8, 1)

        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = MLFC(n_filts * 8 + n_filts * 8, n_filts * 8, k=2, inv_fctr=3)
        self.cnv62 = MLFC(n_filts * 8, n_filts * 8, k=2, inv_fctr=3)
        self.ca5 = ChannelAttention(n_filts * 8)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = MLFC(n_filts * 4 + n_filts * 4, n_filts * 4, k=3, inv_fctr=3)
        self.cnv72 = MLFC(n_filts * 4, n_filts * 4, k=3, inv_fctr=3)
        self.ca6 = ChannelAttention(n_filts * 4)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = MLFC(n_filts * 2 + n_filts * 2, n_filts * 2, k=3, inv_fctr=3)
        self.cnv82 = MLFC(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)
        self.ca7 = ChannelAttention(n_filts * 2)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = MLFC(n_filts + n_filts, n_filts, k=3, inv_fctr=3)
        self.cnv92 = MLFC(n_filts, n_filts, k=3, inv_fctr=3)
        self.ca8 = ChannelAttention(n_filts)
        self.cnv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.cnv2 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.cnv3 = nn.Conv2d(67, 3, kernel_size=1, stride=1, padding=0)
        self.cnv4 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0)
        self.cnv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.cnv6 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.cnv7 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.cnv01 = nn.Conv2d(35, 32, kernel_size=1, stride=1, padding=0)
        self.cnv02 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=(1, 1))
            self.last_activation = None

    def forward(self, x):

        x1 = x
        #IIE Module
        x00 = self.cnv1(x1)
        x01 = self.cnv2(x1)
        x02 = torch.cat([x00,x01,x1],dim=1)
        x03 = self.cnv3(x02)

        x2 = self.cnv11(x03)
        x2p = self.pool(x2)

        x04 = self.pool1(torch.cat([x00,x01],dim=1))
        x2p = self.cnv4(torch.cat([x04,x2p],dim=1))
        x3 = self.cnv21(x2p)
        x3 = self.ca1(x3) * x3
        x3 = self.sa(x3) * x3
        x31 = torch.cat([x2p,x3],dim=1)
        x32 = self.cnv02(x31)
        x3p = self.pool(x3)

        x05 = self.pool1(x04)
        x3p = self.cnv5(torch.cat([x05,x3p],dim=1))
        x4 = self.cnv31(x3p)
        x4 = self.ca2(x4) * x4
        x4 = self.sa(x4) * x4

        x4p = self.pool(x4)
        x06 = self.pool1(x05)
        x4p = self.cnv6(torch.cat([x06,x4p],dim=1))
        x5 = self.cnv41(x4p)
        x5 = self.ca3(x5) * x5
        x5 = self.sa(x5) * x5

        x5p = self.pool(x5)
        x07 = self.pool1(x06)
        x5p = self.cnv7(torch.cat([x07,x5p],dim=1))
        x6 = self.cnv51(x5p)
        x6 = self.ca4(x6) * x6
        x6 = self.sa(x6) * x6

        x2 = self.rspth1(x2)
        x32 = self.rspth2(x32)
        x4 = self.rspth3(x4)
        x5 = self.rspth4(x5)

        x7 = self.up6(x6)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        x7 = self.ca5(x7) * x7
        x7 = self.sa(x7) * x7

        x8 = self.up7(x7)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        x8 = self.ca6(x8) * x8
        x8 = self.sa(x8) * x8

        x9 = self.up8(x8)
        x9 = self.cnv81(torch.cat([x9, x32], dim=1))
        x9 = self.ca7(x9) * x9
        x9 = self.sa(x9) * x9

        x10 = self.up9(x9)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.ca8(x10) * x10
        x10 = self.sa(x10) * x10

        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))

        else:
            logits = self.out(x10)

        return logits
