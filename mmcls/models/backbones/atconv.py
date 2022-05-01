import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmcls.models.builder import MODELS


@MODELS.register_module()
class ATCNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.stem = nn.Conv2d(3, 64, 7, 2, 3)
        self.conv_1 = self.make_downsample_conv(64, 768)
        self.conv_2 = self.make_downsample_conv(128, 1536)
        self.conv_3 = self.make_downsample_conv(256, 1536)
        # self.conv_4 = self.make_downsample_conv(768, 768)

    def make_downsample_conv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, padding=1),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, 3, padding=1),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, 1, stride=2, bias=False))

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_1(x)
        x = self.attention(x, 768)
        x = self.conv_2(x)
        x = self.attention(x, 1536)
        x = self.conv_3(x)
        x = self.attention(x, 1536)
        return x

    def attention(self, x, in_planes):
        row_q = x[:, :in_planes // 6, :, :]
        row_k = x[:, in_planes // 6:in_planes * 2 // 6, :, :]
        row_v = x[:, in_planes * 2 // 6:in_planes * 3 // 6, :, :]
        score_row = torch.einsum("bchw,bcwh->bch",
                                 [row_q, row_k]).softmax(-1).unsqueeze(3)
        row_v = row_v + row_v * score_row
        col_q = x[:, in_planes * 3 // 6:in_planes * 4 // 6, :, :]
        col_k = x[:, in_planes * 4 // 6:in_planes * 5 // 6, :, :]
        col_v = x[:, in_planes * 5 // 6:, :, :]
        score_col = torch.einsum("bchw,bchw->bcw",
                                 [col_q, col_k]).softmax(-1).unsqueeze(2)
        col_v = col_v + col_v * score_col
        return row_v + col_v
