#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        batch_norm=False,
    ):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        layers = []

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()
        layers.extend([conv1, chomp1, relu1])
        if batch_norm:
            batchnorm1 = nn.BatchNorm1d(out_channels)
            layers.append(batchnorm1)

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
        layers.extend([conv2, chomp2, relu2])
        if batch_norm:
            batchnorm2 = nn.BatchNorm1d(out_channels)
            layers.append(batchnorm2)

        # Causal network
        self.causal = torch.nn.Sequential(*layers)

        # Residual connection
        self.upordownsample = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        return out_causal + res


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, *channels, kernel_size=3):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for c_in, c_out in zip(channels, channels[1:]):
            layers.append(
                CausalConvolutionBlock(c_in, c_out, kernel_size, dilation_size)
            )
            dilation_size *= 2  # Doubles the dilation size at each step

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNNEncoder(torch.nn.Module):
    """
    Encode batches of multivariate time-series using a CausalCNN
    Custom average pooling to possibly encode time-series of different lengths that
    were zero-padded on the right (using `torch.nn.utils.rnn.pad_sequence`).
    """

    def __init__(self, *channels_cnn, dim_output_embedding=32, kernel_size=6):
        super().__init__()
        self.causal_cnn = CausalCNN(*channels_cnn, kernel_size=kernel_size)
        self.linear = torch.nn.Linear(channels_cnn[-1], dim_output_embedding)

    def forward(self, x, length=None):
        x = x.transpose(-1, -2)
        h = self.causal_cnn(x)
        if length is not None:
            mask = torch.ones_like(h, requires_grad=False)
            for i in range(mask.shape[0]):
                mask[i, :, length[i] :] = 0
            h = h * mask
            h = h.sum(dim=2)
            h /= length.unsqueeze(1)
        else:
            h = h.mean(dim=-1)
        return self.linear(h)
