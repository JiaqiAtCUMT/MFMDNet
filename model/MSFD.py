import torch.nn as nn
import torch

class decoder_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder_block, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x, x_skip):
        x1 = self.up(x)
        x2 = torch.cat((x1, x_skip), dim=1)
        out = self.up_conv(x2)
        return out


class MSFD_block_16s(nn.Module):
    def __init__(self, class_num):
        super(MSFD_block_16s, self).__init__()
        channel_size = [16, 32, 64, 128, 256, 512, 1024]
        self.de_5 = decoder_block(channel_size[5], channel_size[4])
        self.de_4 = decoder_block(channel_size[4], channel_size[3])
        self.de_3 = decoder_block(channel_size[3], channel_size[2])
        self.de_2 = decoder_block(channel_size[2], channel_size[1])
        self.de_1 = nn.Conv2d(channel_size[1], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_s_8s, x_s_4s, x_s_2s, x_s_1s):
        x5 = self.de_5(x, x_s_8s)
        x4 = self.de_4(x5, x_s_4s)
        x3 = self.de_3(x4, x_s_2s)
        x2 = self.de_2(x3, x_s_1s)
        out = self.de_1(x2)
        return out, x5, x4, x3, x2

class MSFD_block_8s(nn.Module):
    def __init__(self, class_num):
        super(MSFD_block_8s, self).__init__()
        channel_size = [16, 32, 64, 128, 256, 512, 1024]
        self.conv1x1 = nn.Conv2d(channel_size[5], channel_size[4], kernel_size=1, stride=1, padding=0)
        self.de_4 = decoder_block(channel_size[4], channel_size[3])
        self.de_3 = decoder_block(channel_size[3], channel_size[2])
        self.de_2 = decoder_block(channel_size[2], channel_size[1])
        self.de_1 = nn.Conv2d(channel_size[1], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_s_8s, x_s_4s, x_s_2s, x_s_1s):
        x = torch.cat((x, x_s_8s), dim=1)
        x = self.conv1x1(x)

        x4 = self.de_4(x, x_s_4s)
        x3 = self.de_3(x4, x_s_2s)
        x2 = self.de_2(x3, x_s_1s)
        out = self.de_1(x2)
        return out, x4, x3, x2

class MSFD_block_4s(nn.Module):
    def __init__(self, class_num):
        super(MSFD_block_4s, self).__init__()
        channel_size = [16, 32, 64, 128, 256, 512, 1024]
        self.conv1x1 = nn.Conv2d(channel_size[4], channel_size[3], kernel_size=1, stride=1, padding=0)
        self.de_3 = decoder_block(channel_size[3], channel_size[2])
        self.de_2 = decoder_block(channel_size[2], channel_size[1])
        self.de_1 = nn.Conv2d(channel_size[1], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_s_4s, x_s_2s, x_s_1s):
        x = torch.cat((x, x_s_4s), dim=1)
        x = self.conv1x1(x)

        x3 = self.de_3(x, x_s_2s)
        x2 = self.de_2(x3, x_s_1s)
        out = self.de_1(x2)
        return out, x3, x2

class MSFD_block_2s(nn.Module):
    def __init__(self, class_num):
        super(MSFD_block_2s, self).__init__()
        channel_size = [16, 32, 64, 128, 256, 512, 1024]
        self.conv1x1 = nn.Conv2d(channel_size[3], channel_size[2], kernel_size=1, stride=1, padding=0)
        self.de_2 = decoder_block(channel_size[2], channel_size[1])
        self.de_1 = nn.Conv2d(channel_size[1], class_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_s_2s, x_s_1s):
        x = torch.cat((x, x_s_2s), dim=1)
        x = self.conv1x1(x)

        x2 = self.de_2(x, x_s_1s)
        out = self.de_1(x2)
        return out


class MSFD(nn.Module):
    def __init__(self, out_ch):
        super(MSFD, self).__init__()

        self.MSDF_block_16s = MSFD_block_16s(out_ch)
        self.MSDF_block_8s = MSFD_block_8s(out_ch)
        self.MSDF_block_4s = MSFD_block_4s(out_ch)
        self.MSDF_block_2s = MSFD_block_2s(out_ch)

    def forward(self, x, x_s_8s, x_s_4s, x_s_2s, x_s_1s):
        out_m1, x_m1_8s,  x_m1_4s, x_m1_2s, x_m1_1s= self.MSDF_block_16s(x, x_s_8s, x_s_4s, x_s_2s, x_s_1s)
        out_m2, x_m2_4s, x_m2_2s, x_m2_1s = self.MSDF_block_8s(x_s_8s, x_m1_8s, x_m1_4s, x_m1_2s, x_m1_1s)
        out_m3, x_m3_2s, x_m3_1s = self.MSDF_block_4s(x_s_4s, x_m2_4s, x_m2_2s, x_m2_1s)
        out_m4 = self.MSDF_block_2s(x_s_2s, x_m3_2s, x_m3_1s)

        return out_m1, out_m2, out_m3, out_m4
