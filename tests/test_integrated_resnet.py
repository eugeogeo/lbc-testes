import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet18

from integrated_resnet import (
    create_integrated_model, create_integrated_model_scratch,
    inject_texture_filters, make_texture_filters,
)
from train_resnet_scratch_cv import create_scratch_model


class IntegratedResNetTests(unittest.TestCase):
    def test_texture_filters_are_deterministic_normalized_and_structured(self):
        torch.manual_seed(1)
        filters = make_texture_filters()
        torch.manual_seed(100)
        self.assertTrue(torch.equal(filters, make_texture_filters()))
        self.assertEqual(filters.shape, (64, 3, 7, 7))
        self.assertTrue(torch.isfinite(filters).all())
        self.assertTrue(torch.equal(filters[:, 0], filters[:, 1]))
        self.assertTrue(torch.equal(filters[:, 1], filters[:, 2]))
        spatial = filters.sum(dim=1)
        torch.testing.assert_close(spatial.sum(dim=(1, 2)), torch.zeros(64), atol=1e-6, rtol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(spatial, dim=(1, 2)), torch.ones(64))
        for index in range(32, 64, 2):
            self.assertEqual(torch.count_nonzero(spatial[index]).item(), 2)
            torch.testing.assert_close(spatial[index], -spatial[index + 1])
            self.assertLess(spatial[index, 3, 3].item(), 0)

    def test_injection_preserves_other_weights_and_enables_gradients(self):
        model = resnet18(weights=None, num_classes=10)
        model.conv1.weight.requires_grad_(False)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parameter = model.conv1.weight
        result = inject_texture_filters(model)
        self.assertIs(result, model)
        self.assertIs(model.conv1.weight, parameter)
        self.assertTrue(model.conv1.weight.requires_grad)
        torch.testing.assert_close(model.conv1.weight, make_texture_filters())
        self.assertFalse(torch.equal(before["conv1.weight"], model.conv1.weight))
        for name, value in model.state_dict().items():
            if name != "conv1.weight":
                self.assertTrue(torch.equal(before[name], value), name)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        initial = model.conv1.weight.detach().clone()
        loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 32, 32)), torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertGreater(torch.linalg.vector_norm(model.conv1.weight.grad).item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.conv1.weight))

    def test_factory_reuses_pretrained_model_constructor(self):
        model = resnet18(weights=None, num_classes=10)
        with patch("integrated_resnet.create_model", return_value=model) as constructor:
            result = create_integrated_model(10)
        constructor.assert_called_once_with(10)
        self.assertIs(result, model)
        torch.testing.assert_close(result.conv1.weight, make_texture_filters())
        with self.assertRaises(ValueError):
            create_integrated_model(1)

    def test_rejects_incompatible_conv1(self):
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, 3)
        with self.assertRaisesRegex(ValueError, "conv1 deve ter pesos"):
            inject_texture_filters(model)

    def test_scratch_model_changes_only_conv1_from_scratch_baseline(self):
        torch.manual_seed(42)
        original = create_scratch_model(10)
        torch.manual_seed(42)
        with patch("integrated_resnet.resnet18", wraps=resnet18) as constructor:
            model = create_integrated_model_scratch(10)
        constructor.assert_called_once_with(weights=None)
        self.assertEqual(model.fc.out_features, 10)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        torch.testing.assert_close(model.conv1.weight, make_texture_filters())
        self.assertFalse(torch.equal(model.conv1.weight, original.conv1.weight))
        for name, value in model.state_dict().items():
            if name != "conv1.weight":
                self.assertTrue(torch.equal(value, original.state_dict()[name]), name)
        with self.assertRaises(ValueError):
            create_integrated_model_scratch(1)


if __name__ == "__main__":
    unittest.main()
