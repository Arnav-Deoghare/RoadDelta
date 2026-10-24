"""Basic change-detection model for RoadDelta: a small U-Net with early fusion.

Input  : old image + new image stacked on the channel axis -> (B, 6, H, W)
         (add the rasterized old road map as a 7th channel later: in_channels=7)
Output : logits (B, 2, H, W) -> channel 0 = road added, channel 1 = road removed
H and W must be divisible by 4 (two downsampling steps).
"""
import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class ChangeUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, base=32):
        super().__init__()
        self.enc1 = block(in_channels, base)
        self.enc2 = block(base, base * 2)
        self.bottom = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, old_img, new_img, extra=None):
        parts = [old_img, new_img] + ([extra] if extra is not None else [])
        x = torch.cat(parts, dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    model = ChangeUNet()
    old = torch.randn(2, 3, 256, 256)
    new = torch.randn(2, 3, 256, 256)
    out = model(old, new)
    print("output:", tuple(out.shape), "| params:", sum(p.numel() for p in model.parameters()))