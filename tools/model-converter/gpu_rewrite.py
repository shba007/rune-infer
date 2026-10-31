"""Optional export-only rank-4 linear lowering for LiteRT GPU experiments."""
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


if __name__ == "__main__":
    check_rewrite()
