import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, channel_lst=[32, 64, 128]):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=channel_lst[0],
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=channel_lst[0],
                                 out_channels=channel_lst[1],
                                 kernel_size=3,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv2d(in_channels=channel_lst[1],
                                 out_channels=channel_lst[1],
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._conv_4 = nn.Conv2d(in_channels=channel_lst[1],
                                 out_channels=channel_lst[2],
                                 kernel_size=3,
                                 stride=2, padding=1)

        self._residual_stack_1 = ResidualStack(in_channels=channel_lst[1],
                                               num_hiddens=channel_lst[1],
                                               num_residual_layers=num_residual_layers,
                                               num_residual_hiddens=channel_lst[0])
        self._residual_stack_2 = ResidualStack(in_channels=channel_lst[2],
                                               num_hiddens=channel_lst[2],
                                               num_residual_layers=num_residual_layers,
                                               num_residual_hiddens=channel_lst[1])

        self._liner = nn.Linear(4 * 4 * channel_lst[2], num_hiddens)

        self.batch_norm1 = nn.BatchNorm2d(channel_lst[0])
        self.batch_norm2 = nn.BatchNorm2d(channel_lst[1])
        self.batch_norm3 = nn.BatchNorm2d(channel_lst[1])
        self.batch_norm4 = nn.BatchNorm2d(channel_lst[1])
        self.batch_norm5 = nn.BatchNorm2d(channel_lst[2])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self.batch_norm1(x)
        x = self.relu(x)    # 16 x 16

        x = self._conv_2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)    # 8 x 8

        x = self._conv_3(x)
        x = self.batch_norm3(x) # 8 x 8

        x = self._residual_stack_1(x)
        x = self.batch_norm4(x)
        x = self.relu(x)

        x = self._conv_4(x)
        x = self.batch_norm5(x) # 4 x 4

        x = self._residual_stack_2(x)

        x_flat = torch.flatten(x, start_dim=1)

        features = self._liner(x_flat)

        return features


class Decoder(nn.Module):
    def __init__(self, in_channels, num_residual_layers, num_residual_hiddens, channel_lst=[32, 64, 128]):
        super(Decoder, self).__init__()

        self._liner = nn.Linear(in_channels, 4 * 4 * channel_lst[-1])

        self._residual_stack = ResidualStack(in_channels=channel_lst[-3],
                                              num_hiddens=channel_lst[-3],
                                              num_residual_layers=num_residual_layers,
                                              num_residual_hiddens=channel_lst[-3] // 2)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=channel_lst[-1],
                                                out_channels=channel_lst[-2],
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=channel_lst[-2],
                                                out_channels=channel_lst[-3],
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=channel_lst[-3],
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(channel_lst[-2])
        self.batch_norm2 = nn.BatchNorm2d(channel_lst[-3])
        self.batch_norm3 = nn.BatchNorm2d(channel_lst[-3])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self._liner(inputs)
        x = x.view([-1, 128, 4, 4])

        x = self._conv_trans_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self._conv_trans_2(x)
        x = self.batch_norm2(x)

        x = self._residual_stack(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        return self._conv_trans_3(x)


class AE(nn.Module):
    def __init__(self, num_residual_layers, num_residual_hiddens, embedding_dim, channel_lst=[32, 64, 128]):
        super(AE, self).__init__()

        self.embedding_dim = embedding_dim

        self._encoder = Encoder(3, embedding_dim,
                                num_residual_layers,
                                num_residual_hiddens,
                                channel_lst)
        self._decoder = Decoder(embedding_dim,
                                num_residual_layers,
                                num_residual_hiddens,
                                channel_lst)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        features = self._encoder(x)
        x_recon = self._decoder(features)

        return x_recon, features
