from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from model.MSFD import MSFD
from model.MMFI import MMFI

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_d(nn.Module):
    def __init__(self, d_ch, out_ch):
        super(conv_block_d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(d_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, y):
        y = self.conv(y)
        return y


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x):
        e1 = self.Conv1(x)  # batchsize*32*512*512

        e2 = self.Maxpool1(e1)  # 32*256*256
        e2 = self.Conv2(e2)  # 64*256*256

        e3 = self.Maxpool2(e2)  # 64*128*128
        e3 = self.Conv3(e3)  # 128*128*128

        e4 = self.Maxpool3(e3)  # 128*64*64
        e4 = self.Conv4(e4)  # 256*64*64

        e5 = self.Maxpool4(e4)  # 256*32*32
        e5 = self.Conv5(e5)  # 512*32*32

        return e5, e4, e3, e2, e1


class Dual_Encoder(nn.Module):
    """
    Encoder has two branch, and the fusion mechanism is adding from the depth branch into the RGB branch
    """
    def __init__(self, in_ch, d_ch):
        super(Dual_Encoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

    def forward(self, x, y):
        e1 = self.Conv1(x)  # batchsize*64*1024*1024
        t1 = self.Conv1_d(y)  # 64*1024*1024
        e1 = e1 + t1  # fuse 64*1024*1024

        e2 = self.Maxpool1(e1)  # 64*512*512
        e2 = self.Conv2(e2)  # 128*512*512
        t2 = self.Maxpool1_d(t1)  # 64*512*512
        t2 = self.Conv2_d(t2)  # 128*512*512
        e2 = e2 + t2  # fuse 128*512*512

        e3 = self.Maxpool2(e2)  # 128*256*256
        e3 = self.Conv3(e3)  # 256*256*256
        t3 = self.Maxpool2_d(t2)  # 128*256*256
        t3 = self.Conv3_d(t3)  # 256*256*256
        e3 = e3 + t3  # fuse 256*256*256

        e4 = self.Maxpool3(e3)  # 256*128*128
        e4 = self.Conv4(e4)  # 512*128*128
        t4 = self.Maxpool3_d(t3)
        t4 = self.Conv4_d(t4)
        e4 = e4 + t4  # 512*128*128

        e5 = self.Maxpool4(e4)  # 512*64*64
        e5 = self.Conv5(e5)  # 1024*64*64
        t5 = self.Maxpool4_d(t4)
        t5 = self.Conv5_d(t5)
        e5 = e5 + t5  # 1024*64*64

        return e5, e4, e3, e2, e1


class Dual_fusion_Encoder(nn.Module):
    def __init__(self, in_ch, d_ch):
        super(Dual_fusion_Encoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

        self.mmfi_1 = MMFI(filters[0])
        self.mmfi_2 = MMFI(filters[1])
        self.mmfi_3 = MMFI(filters[2])
        self.mmfi_4 = MMFI(filters[3])
        self.mmfi_5 = MMFI(filters[4])

    def forward(self, x, y):
        x1 = self.Conv1(x)
        y1 = self.Conv1_d(y)

        xy1_fusion_map = self.mmfi_1(x1, y1)
        x1 = x1 + xy1_fusion_map
        y1 = y1 + xy1_fusion_map

        # the second blcok
        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)
        y2 = self.Maxpool1_d(y1)
        y2 = self.Conv2_d(y2)

        xy2_fusion_map = self.mmfi_2(x2, y2)
        x2 = x2 + xy2_fusion_map
        y2 = y2 + xy2_fusion_map

        # the third block
        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)
        y3 = self.Maxpool2_d(y2)
        y3 = self.Conv3_d(y3)

        xy3_fusion_map = self.mmfi_3(x3, y3)
        x3 = x3 + xy3_fusion_map
        y3 = y3 + xy3_fusion_map

        # the forth block
        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)
        y4 = self.Maxpool3_d(y3)
        y4 = self.Conv4_d(y4)

        xy4_fusion_map = self.mmfi_4(x4, y4)
        x4 = x4 + xy4_fusion_map
        y4 = y4 + xy4_fusion_map

        # the fifth block(between the encoder and the decoder)
        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)
        y5 = self.Maxpool4_d(y4)
        y5 = self.Conv5_d(y5)

        xy5_fusion_map = self.mmfi_5(x5, y5)
        x5 = x5 + xy5_fusion_map

        return x5, x4, x3, x2, x1


class Dual_fusion_Encoder_mid(nn.Module):
    def __init__(self, in_ch, d_ch):
        super(Dual_fusion_Encoder_mid, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

        self.mmfi_1 = MMFI(filters[0])
        self.mmfi_2 = MMFI(filters[1])
        self.mmfi_3 = MMFI(filters[2])
        self.mmfi_4 = MMFI(filters[3])
        self.mmfi_5 = MMFI(filters[4])

    def forward(self, x, y):
        x1 = self.Conv1(x)
        y1 = self.Conv1_d(y)

        xy1_early = torch.mean(torch.cat((x1, y1), dim=1), dim=1)
        xy1_fusion_map = self.mmfi_1(x1, y1)
        xy1_late = torch.mean(xy1_fusion_map, dim=1)
        x1 = x1 + xy1_fusion_map
        y1 = y1 + xy1_fusion_map

        # the second blcok
        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)
        y2 = self.Maxpool1_d(y1)
        y2 = self.Conv2_d(y2)

        xy2_early = torch.mean(torch.cat((x2, y2), dim=1), dim=1)
        xy2_fusion_map = self.mmfi_2(x2, y2)
        xy2_late = torch.mean(xy2_fusion_map, dim=1)
        x2 = x2 + xy2_fusion_map
        y2 = y2 + xy2_fusion_map

        # the third block
        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)
        y3 = self.Maxpool2_d(y2)
        y3 = self.Conv3_d(y3)

        xy3_early = torch.mean(torch.cat((x3, y3), dim=1), dim=1)
        xy3_fusion_map = self.mmfi_3(x3, y3)
        xy3_late = torch.mean(xy3_fusion_map, dim=1)
        x3 = x3 + xy3_fusion_map
        y3 = y3 + xy3_fusion_map

        # the forth block
        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)
        y4 = self.Maxpool3_d(y3)
        y4 = self.Conv4_d(y4)

        xy4_early = torch.mean(torch.cat((x4, y4), dim=1), dim=1)
        xy4_fusion_map = self.mmfi_4(x4, y4)
        xy4_late = torch.mean(xy4_fusion_map, dim=1)
        x4 = x4 + xy4_fusion_map
        y4 = y4 + xy4_fusion_map

        # the fifth block(between the encoder and the decoder)
        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)
        y5 = self.Maxpool4_d(y4)
        y5 = self.Conv5_d(y5)

        xy5_early = torch.mean(torch.cat((x5, y5), dim=1), dim=1)
        xy5_fusion_map = self.mmfi_5(x5, y5)
        xy5_late = torch.mean(xy5_fusion_map, dim=1)
        x5 = x5 + xy5_fusion_map

        return x5, x4, x3, x2, x1, xy1_early, xy1_late, xy2_early, xy2_late, xy3_early, xy3_late, xy4_early, xy4_late, xy5_early, xy5_late


class Decoder(nn.Module):
    def __init__(self, out_ch=5):
        super(Decoder, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, e5, e4, e3, e2, e1):
        d5 = self.Up5(e5)  # 512*128*128
        d5 = torch.cat((e4, d5), dim=1)  # 1024*128*128
        d5 = self.Up_conv5(d5)  # 512*128*128

        d4 = self.Up4(d5)  # 256*256*256
        d4 = torch.cat((e3, d4), dim=1)  # 512*256*256
        d4 = self.Up_conv4(d4)  # 256*256*256

        d3 = self.Up3(d4)  # 128*512*512
        d3 = torch.cat((e2, d3), dim=1)  # 256*512*512
        d3 = self.Up_conv3(d3)  # 128*512*512

        d2 = self.Up2(d3)  # 64*1024*1024
        d2 = torch.cat((e1, d2), dim=1)  # 128*1024*1024
        d2 = self.Up_conv2(d2)  # 64*1024*1024

        out = self.Conv(d2)  # 5*1024*1024

        return out


class U_Net(nn.Module):

    def __init__(self, in_ch=3, out_ch=5):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #depth
        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)


        # self.conv_test = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):    # x:batchsize*3*512*512

        e1 = self.Conv1(x)      #batchsize*32*512*512

        e2 = self.Maxpool1(e1)  #32*256*256
        e2 = self.Conv2(e2)     #64*256*256

        e3 = self.Maxpool2(e2)  #64*128*128
        e3 = self.Conv3(e3)     #128*128*128

        e4 = self.Maxpool3(e3)  #128*64*64
        e4 = self.Conv4(e4)     #256*64*64

        e5 = self.Maxpool4(e4)  # 256*32*32
        e5 = self.Conv5(e5)     # 512*32*32

        d5 = self.Up5(e5)       # 512*64*64
        d5 = torch.cat((e4, d5), dim=1)  # 1024*64*64
        d5 = self.Up_conv5(d5)  # 512*64*64

        d4 = self.Up4(d5)       # 256*128*128
        d4 = torch.cat((e3, d4), dim=1)  # 512*128*128
        d4 = self.Up_conv4(d4)  # 256*128*128

        d3 = self.Up3(d4)       # 128*256*256
        d3 = torch.cat((e2, d3), dim=1)  # 256*256*256
        d3 = self.Up_conv3(d3)  # 128*256*256

        d2 = self.Up2(d3)       # 64*512*512
        d2 = torch.cat((e1, d2), dim=1) # 128*512*512
        d2 = self.Up_conv2(d2)  # 64*512*512

        out = self.Conv(d2)     # 5*512*512

        # d1 = self.active(out)

        return out

class Dual_U_Net(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=1):
        super(Dual_U_Net, self).__init__()

        self.encoder = Dual_Encoder(in_ch, d_ch)
        self.decoder = Decoder(out_ch)

    def forward(self, x, y):
        x5, x4, x3, x2, x1 = self.encoder(x, y)
        out = self.decoder(x5, x4, x3, x2, x1)
        return out


class U_Net_MSDF(nn.Module):
    def __init__(self, in_ch=3, out_ch=5):
        super(U_Net_MSDF, self).__init__()
        self.encoder = Encoder(in_ch)
        self.msfd = MSFD(out_ch)

    def forward(self, x):
        x_16s, x_8s, x_4s, x_2s, x_1s = self.encoder(x)
        out_1, out_2, out_3, out_4 = self.msfd(x_16s, x_8s, x_4s, x_2s, x_1s)

        return [out_1, out_2, out_3, out_4]


class Dual_UNet_MSDF_MMFI(nn.Module):
    """
        Encoder: two branch
        Fusion : MMFI
        Decoder: MSFD
    """
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(Dual_UNet_MSDF_MMFI, self).__init__()
        self.encoder = Dual_fusion_Encoder(in_ch, d_ch)
        self.msfd = MSFD(out_ch)

    def forward(self, x, y):
        x_16s, x_8s, x_4s, x_2s, x_1s = self.encoder(x, y)
        out_1, out_2, out_3, out_4 = self.msfd(x_16s, x_8s, x_4s, x_2s, x_1s)
        return [out_1, out_2, out_3, out_4]


class Dual_UNet_MSDF(nn.Module):
    """
        Encoder: two branch
        Fusion : adding from the depth into the RGB
        Decoder: MSFD
    """
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(Dual_UNet_MSDF, self).__init__()
        self.encoder = Dual_Encoder(in_ch, d_ch)
        self.msfd = MSFD(out_ch)

    def forward(self, x, y):
        x5, x4, x3, x2, x1 = self.encoder(x, y)
        out_1, out_2, out_3, out_4 = self.msfd(x5, x4, x3, x2, x1)
        return [out_1, out_2, out_3, out_4]


class Dual_UNet_MMFI(nn.Module):
    """
        Encoder: two branch
        Fusion : MMFI
        Decoder: one stream decoder
    """
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(Dual_UNet_MMFI, self).__init__()
        self.encoder = Dual_fusion_Encoder(in_ch, d_ch)
        self.decoder = Decoder(out_ch)

    def forward(self, x, y):
        x5, x4, x3, x2, x1 = self.encoder(x, y)
        out = self.decoder(x5, x4, x3, x2, x1)
        return out
