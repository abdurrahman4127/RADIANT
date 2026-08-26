import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    """
    two convolution stages, each followed by instance normalization
    and a leaky relu activation
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class UNet3D(nn.Module):
    """
    symmetric three level 3d u-net with a shared bottleneck used for
    segmentation, radiomics regression, and domain discrimination
    """

    def __init__(self, in_ch=4, base_ch=16, out_ch=3):
        super().__init__()

        self.enc1 = conv_block(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(
            base_ch * 8, base_ch * 4, kernel_size=2, stride=2
        )
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(
            base_ch * 4, base_ch * 2, kernel_size=2, stride=2
        )
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(
            base_ch * 2, base_ch, kernel_size=2, stride=2
        )
        self.dec1 = conv_block(base_ch * 2, base_ch)
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)

        self._init_weights()
    

    def _init_weights(self):
        """
        applies kaiming normal initialization to convolution weights
        and unit scale, zero shift to normalization layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out_logits = self.out_conv(d1)
        out_prob = torch.sigmoid(out_logits)
        return out_prob, b


def encoder_parameters(model):
    """
    returns the parameter list belonging to the encoder and
    bottleneck, used to apply a reduced learning rate during
    adaptation
    """
    return (
        list(model.enc1.parameters())
        + list(model.enc2.parameters())
        + list(model.enc3.parameters())
        + list(model.bottleneck.parameters())
    )


def decoder_parameters(
    model,
):
    """
    returns every named parameter that does not belong to the
    encoder or bottleneck.
    """
    encoder_names = (
        "enc1",
        "enc2",
        "enc3",
        "bottleneck",
    )
    
    return [
        p for n, p in model.named_parameters()
        if not n.startswith(encoder_names)
    ]


class RadRegressor(nn.Module):
    """
    two layer multilayer perceptron mapping pooled bottleneck
    features to a predicted radiomics vector
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden = max(in_dim // 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


"""
reduces a spatial feature map to a single vector per sample
through global average pooling
"""
def global_pool(feat):
    return F.adaptive_avg_pool3d(feat, output_size=1).view(feat.size(0), -1)


class GradReverse(torch.autograd.Function):
    """
    identity mapping during the forward pass, gradient negation and
    scaling during the backward pass
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """
    three layer fully connected network with decreasing width that
    classifies pooled bottleneck features as source or target domain
    """

    def __init__(self, in_dim):
        super().__init__()
        h1 = max(in_dim // 2, 16)
        h2 = max(in_dim // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, grl_alpha=1.0):
        x = x.view(x.size(0), -1)
        x = GradReverse.apply(x, grl_alpha)
        return self.net(x)
