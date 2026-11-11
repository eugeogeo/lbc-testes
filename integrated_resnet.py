"""Inicialização estrutural de conv1 com filtros de textura determinísticos."""

import math

import torch
from torch import nn
from torchvision.models import resnet18

from resnet_baseline import create_model


GABOR_WAVELENGTHS = (2.0, 3.0, 4.0, 6.0)
GABOR_ORIENTATIONS = tuple(index * math.pi / 8 for index in range(8))
LBP_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))


def make_texture_filters():
    """Gera 32 filtros Gabor e 32 diferenças locais, no formato [64, 3, 7, 7]."""
    coordinates = torch.arange(-3, 4, dtype=torch.float64)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    filters = []

    for wavelength in GABOR_WAVELENGTHS:
        for theta in GABOR_ORIENTATIONS:
            x_rot = x * math.cos(theta) + y * math.sin(theta)
            y_rot = -x * math.sin(theta) + y * math.cos(theta)
            envelope = torch.exp(-(x_rot.square() + 0.5**2 * y_rot.square()) / (2 * 2.0**2))
            kernel = envelope * torch.cos(2 * math.pi * x_rot / wavelength)
            kernel = kernel - kernel.mean()
            filters.append(kernel / torch.linalg.vector_norm(kernel))

    for radius in (1, 3):
        for row_offset, col_offset in LBP_OFFSETS:
            kernel = torch.zeros((7, 7), dtype=torch.float64)
            kernel[3, 3] = -1
            kernel[3 + radius * row_offset, 3 + radius * col_offset] = 1
            kernel = kernel / torch.linalg.vector_norm(kernel)
            filters.extend((kernel, -kernel))

    spatial_filters = torch.stack(filters).to(torch.float32)
    return spatial_filters.unsqueeze(1).repeat(1, 3, 1, 1) / 3


def inject_texture_filters(model):
    """Substitui conv1.weight via state_dict e mantém a camada treinável."""
    weight = model.conv1.weight
    expected_shape = (64, 3, 7, 7)
    if tuple(weight.shape) != expected_shape or model.conv1.bias is not None:
        raise ValueError(f"conv1 deve ter pesos {expected_shape} e não ter bias.")

    filters = make_texture_filters().to(device=weight.device, dtype=weight.dtype)
    model.conv1.load_state_dict({"weight": filters}, strict=True)
    model.conv1.weight.requires_grad_(True)
    return model


def create_integrated_model(num_classes=10):
    """Cria ResNet-18 pré-treinada com fc adaptada e conv1 inicializada por textura."""
    if num_classes < 2:
        raise ValueError("São necessárias pelo menos duas classes.")
    return inject_texture_filters(create_model(num_classes))


def create_integrated_model_scratch(num_classes):
    """Cria ResNet-18 sem pré-treino e injeta filtros de textura em conv1."""
    if num_classes < 2:
        raise ValueError("São necessárias pelo menos duas classes.")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return inject_texture_filters(model)


if __name__ == "__main__":
    model = create_integrated_model(num_classes=10)
    print(f"conv1.weight: {tuple(model.conv1.weight.shape)}")
    print(f"conv1 treinável: {model.conv1.weight.requires_grad}")
    print(f"Número de classes: {model.fc.out_features}")
