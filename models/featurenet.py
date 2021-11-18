import torch
import torch.nn as nn 


class FeatureNet(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], interm_dim=128, out_dim=128):
        super(FeatureNet, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=num_channels[0],
                                 out_channels=interm_dim,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=interm_dim + num_channels[1],
                                 out_channels=interm_dim * 2,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=interm_dim * 2 + num_channels[2],
                                 out_channels=interm_dim * 3,
                                 kernel_size=3,
                                 stride=2, padding=1)

        self._conv_4 = nn.Conv2d(in_channels=interm_dim * 3 + num_channels[3],
                                 out_channels=interm_dim * 4,
                                 kernel_size=3,
                                 stride=2, padding=1)

        self.linear = nn.Linear(4 * interm_dim, out_dim)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, features):
        out1 = self.relu(self._conv_1(features[0]))
        in1 = torch.cat((out1, features[1]), 1)    # 128 + 128

        out2 = self.relu(self._conv_2(in1))
        in2 = torch.cat((out2, features[2]), 1)  # 128 + 128 + 256

        out3 = self.relu(self._conv_3(in2))
        in3 = torch.cat((out3, features[3]), 1)  # 128 + 128 + 128 + 512

        out4 = self.lrelu(self._conv_4(in3)) # 128 + 128 + 128 + 128

        _out = self.GAP(out4)
        _out = _out.view(_out.size(0), -1)

        out = self.linear(_out)

        return out
