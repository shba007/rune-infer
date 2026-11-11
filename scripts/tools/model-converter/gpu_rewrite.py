"""Optional export-only rewrites for LiteRT GPU experiments."""
import torch
from torch.nn import functional as F


class LinearAsConv(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, pixels):
        # Preserve token order while avoiding the delegate's rank-3 FC layout.
        shape = pixels.shape
        channels = shape[-1]
        packed = pixels.reshape(-1, channels).transpose(0, 1)[None, :, None, :]
        result = F.conv2d(packed, self.weight[:, :, None, None], self.bias)
        return result[0, :, 0, :].transpose(0, 1).reshape(*shape[:-1], self.weight.shape[0])


def replace_linear_with_conv(module):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LinearAsConv(child).eval())
            replaced += 1
        else:
            replaced += replace_linear_with_conv(child)
    return replaced


def fold_dinov2_layer_scales(module, expected_layers=24):
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer, Dinov2LayerScale
    pairs = []
    for name, layer in module.named_modules():
        if not isinstance(layer, Dinov2Layer):
            continue
        for scale_name, projection in (
            ("layer_scale1", layer.attention.output.dense),
            ("layer_scale2", layer.mlp.fc2),
        ):
            scale = getattr(layer, scale_name)
            if (layer.training or not isinstance(projection, torch.nn.Linear)
                    or not isinstance(scale, Dinov2LayerScale)
                    or scale.lambda1.shape != (projection.out_features,)
                    or projection.weight.dtype != torch.float32
                    or not torch.isfinite(scale.lambda1).all()):
                raise ValueError(f"Unsupported LayerScale/projection at {name}.{scale_name}")
            pairs.append((name, layer, scale_name, projection, scale.lambda1))
    if len(pairs) != expected_layers * 2:
        raise ValueError(f"Expected {expected_layers * 2} LayerScales, found {len(pairs)}")
    # Validate the complete topology before changing this in-memory export model.
    # In eval mode Dropout between projection and scale is an identity.
    with torch.no_grad():
        for _, layer, scale_name, projection, gamma in pairs:
            projection.weight.mul_(gamma[:, None])
            if projection.bias is not None:
                projection.bias.mul_(gamma)
            setattr(layer, scale_name, torch.nn.Identity())
    return [f"{name}.{scale_name}" for name, _, scale_name, _, _ in pairs]


def check_rewrite():
    torch.manual_seed(0)
    for bias in (False, True):
        linear = torch.nn.Linear(16, 32, bias=bias).eval()
        rewritten = LinearAsConv(linear).eval()
        assert rewritten.weight is linear.weight and rewritten.bias is linear.bias
        for shape in ((16,), (7, 16), (2, 7, 16), (2, 3, 7, 16)):
            pixels = torch.randn(shape)
            with torch.no_grad():
                torch.testing.assert_close(rewritten(pixels), linear(pixels), atol=1e-6, rtol=1e-5)
    nested = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Sequential(torch.nn.Linear(3, 4)))
    assert replace_linear_with_conv(nested) == 2
    print("Linear-to-convolution checks passed: ranks 1–4, bias, shared weights and nested modules.")
    from transformers import Dinov2Config
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer
    config = Dinov2Config(hidden_size=16, num_attention_heads=4, mlp_ratio=2)
    config._attn_implementation = "eager"
    layer = Dinov2Layer(config).eval()
    with torch.no_grad():
        layer.layer_scale1.lambda1.copy_(torch.linspace(-1, 1, 16))
        layer.layer_scale2.lambda1.copy_(torch.linspace(0, 2, 16))
        pixels = torch.randn(2, 7, 16)
        before = layer(pixels)
        names = fold_dinov2_layer_scales(layer, expected_layers=1)
        torch.testing.assert_close(layer(pixels), before, atol=1e-6, rtol=1e-5)
    assert len(names) == 2
    try:
        fold_dinov2_layer_scales(layer, expected_layers=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated folding must fail instead of rescaling weights")
    print("LayerScale folding checks passed: complete transformer block, signed scales and repeated-fold rejection.")


if __name__ == "__main__":
    check_rewrite()
