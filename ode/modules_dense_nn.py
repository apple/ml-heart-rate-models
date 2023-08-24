#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

PersonalizationType = Literal["none", "softmax", "concatenate"]


class DenseNN(torch.nn.Module):
    def __init__(
        self,
        *dim_layers,
        activation: nn.Module = torch.nn.Softplus(),
        dim_context=0,
        bias=True,
        output_bounds=None,
    ):
        super().__init__()
        self.dim_layers = np.array(dim_layers)
        self.dim_context = dim_context

        # concatenate context to the input
        self.dim_layers[0] += self.dim_context
        layers = []
        for i, (l, r) in enumerate(zip(self.dim_layers, self.dim_layers[1:])):
            layers.append(torch.nn.Linear(l, r, bias=bias))
        self.layers = torch.nn.ModuleList(layers)
        self.f = activation
        self.output_bounds = output_bounds

    def forward(self, x, *context):
        if self.dim_context:
            x = torch.cat([x, *context], dim=-1)

        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        if self.output_bounds is not None:
            h = (
                torch.sigmoid(h) * (self.output_bounds[1] - self.output_bounds[0])
                + self.output_bounds[0]
            )

        return h


class PersonalizedScalarNN(torch.nn.Module):
    def __init__(
        self,
        *dim_layers,
        personalization: PersonalizationType,
        dim_personalization=0,
        output_bounds=None,
        activation: nn.Module = torch.nn.Softplus(),
        bias=True,
    ):
        super().__init__()
        self.personalization = personalization
        if self.personalization == "none":
            dim_personalization = 0
            self.dense_nn = DenseNN(
                *dim_layers,
                1,
                dim_context=dim_personalization,
                output_bounds=output_bounds,
                activation=activation,
                bias=bias,
            )
        elif self.personalization == "softmax":
            # add extra
            self.dense_nn = DenseNN(
                *dim_layers,
                dim_personalization,
                dim_context=0,
                output_bounds=output_bounds,
                activation=activation,
            )
        elif self.personalization == "concatenate":
            self.dense_nn = DenseNN(
                *dim_layers,
                1,
                dim_context=dim_personalization,
                output_bounds=output_bounds,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown personalization {personalization}")

    def forward(self, x, context=None):
        if self.personalization == "none":
            return self.dense_nn(x).squeeze(-1)
        elif self.personalization == "softmax":
            h = self.dense_nn(x)
            return (h * torch.softmax(context, dim=-1)).sum(dim=-1)
        elif self.personalization == "concatenate":
            return self.dense_nn(x, context).squeeze(-1)
