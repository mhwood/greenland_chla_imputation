
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Module):
    """
    Mask-aware convolution.

    x:    [B, C, H, W]
    mask: [B, 1, H, W], 1 = valid ocean, 0 = land
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        nn.init.constant_(self.mask_conv.weight, 1.0)

        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        x = x * mask

        out = self.conv(x)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)

        # normalize for number of valid pixels in the kernel
        out = out / mask_sum.clamp_min(1.0)

        # preserve original invalid land mask
        new_mask = (mask_sum > 0).float()

        return out, new_mask

class MaskedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = MaskedConv2d(in_ch, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)

        self.conv2 = MaskedConv2d(out_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.act = nn.GELU()

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = self.act(self.norm1(x))

        x, mask = self.conv2(x, mask)
        x = self.act(self.norm2(x))

        return x, mask


def crop_or_pad(x, ref):
    """
    Makes x match ref spatial dimensions.
    Needed because width = 103 is odd.
    """
    _, _, h, w = x.shape
    _, _, hr, wr = ref.shape

    dh = hr - h
    dw = wr - w

    if dh != 0 or dw != 0:
        x = F.pad(
            x,
            [
                dw // 2,
                dw - dw // 2,
                dh // 2,
                dh - dh // 2,
            ],
        )

    return x[:, :, :hr, :wr]

def downsample_mask(mask):
    """
    Max-pool mask so a coarse cell is ocean if any contributing fine cell is ocean.
    """
    return F.max_pool2d(mask, kernel_size=2, stride=2)


class ChlUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base=64):
        super().__init__()

        self.enc1 = MaskedConvBlock(in_channels, base)
        self.enc2 = MaskedConvBlock(base, base * 2)
        self.enc3 = MaskedConvBlock(base * 2, base * 4)
        self.enc4 = MaskedConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = MaskedConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = MaskedConvBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = MaskedConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = MaskedConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = MaskedConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x, ocean_mask):
        """
        x:          [B, C, 184, 103]
        ocean_mask: [B, 1, 184, 103]
        """

        # enforce no land input
        x = x * ocean_mask

        e1, m1 = self.enc1(x, ocean_mask)

        x2 = self.pool(e1)
        m2 = downsample_mask(m1)
        e2, m2 = self.enc2(x2, m2)

        x3 = self.pool(e2)
        m3 = downsample_mask(m2)
        e3, m3 = self.enc3(x3, m3)

        x4 = self.pool(e3)
        m4 = downsample_mask(m3)
        e4, m4 = self.enc4(x4, m4)

        xb = self.pool(e4)
        mb = downsample_mask(m4)
        b, mb = self.bottleneck(xb, mb)

        d4 = self.up4(b)
        d4 = crop_or_pad(d4, e4)
        d4, md4 = self.dec4(torch.cat([d4, e4], dim=1), m4)

        d3 = self.up3(d4)
        d3 = crop_or_pad(d3, e3)
        d3, md3 = self.dec3(torch.cat([d3, e3], dim=1), m3)

        d2 = self.up2(d3)
        d2 = crop_or_pad(d2, e2)
        d2, md2 = self.dec2(torch.cat([d2, e2], dim=1), m2)

        d1 = self.up1(d2)
        d1 = crop_or_pad(d1, e1)
        d1, md1 = self.dec1(torch.cat([d1, e1], dim=1), m1)

        pred = self.out(d1)

        # force land output to zero
        pred = pred * ocean_mask

        return pred
