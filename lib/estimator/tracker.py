
import torch
import torch.nn as nn

from copy import deepcopy


def weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights,
                             global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights,
                         global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
    # Iterate over all model layers
    for layer_number, layer in enumerate(model.modules()):
        # Compute weight histograms for appropriate layer
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            weight_histograms_conv2d(writer, step, weights, layer_number)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, layer_number)


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = deepcopy(model)

    def forward(self, x):
        out = self.model(x)
        return out
