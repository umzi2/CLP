import warnings

import torch
from torch import nn
from torch.nn import functional as F


class dinov2(nn.Module):
    """
    DINOv2 backend, developed by musl from the neosr-project: https://github.com/neosr-project/neosr
    """

    def __init__(self, layers=None, weights=None, norm=True):
        super().__init__()

        # load model and suppress xformers dependency warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = (
                torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vitb14",
                    trust_repo="check",
                    verbose=False,
                )
                .to("cuda", memory_format=torch.channels_last, non_blocking=True)
                .eval()
            )

        if layers is None:
            layers = [0, 1, 2, 3, 4, 5, 6, 7]
        if weights is None:
            weights = (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.1)

        self.layers = layers
        self.chns = [768] * len(self.layers)

        if len(weights) != len(self.layers):
            msg = "Number of layer weights must match number of layers"
            raise ValueError(msg)

        self.register_buffer(
            "layer_weights", torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
        )

        self.norm = norm
        if self.norm:
            # imagenet norm values
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            )

        for param in self.parameters():
            param.requires_grad = False

    def adapt_size(self, dim):
        return ((dim + 13) // 14) * 14

    def get_features(self, x):
        if self.norm:
            x = (x - self.mean) / self.std
        # pad because embedded patch expects multiples of 14
        _, _, H, W = x.shape
        target_h = self.adapt_size(H)
        target_w = self.adapt_size(W)
        pad_h = target_h - H
        pad_w = target_w - W

        if pad_h != 0 or pad_w != 0:
            x = F.pad(
                x,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="reflect",
            )

        # extract features
        features = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=True, return_class_token=False
        )
        return [
            feat * weight
            for feat, weight in zip(features, self.layer_weights, strict=False)
        ]

    def forward(self, x):
        return self.get_features(x)


def cosine_triplet_loss(anchor, positive, negative, margin=0.1):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    d_pos = 1 - (anchor * positive).sum(dim=-1)
    d_neg = 1 - (anchor * negative).sum(dim=-1)

    return F.relu(d_pos - d_neg + margin).mean()


class CLPLoss(nn.Module):
    def __init__(
        self, loss_weight: float = 1.0, criterion="fd", patch_size=4, num_proj=24
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.model = dinov2()
        self.index = 0
        if criterion == "st":
            self.criterion = cosine_triplet_loss
        elif criterion == "fd":
            self.criterion = self.fd
            for i in range(len(self.model.chns)):
                rand = torch.randn(
                    num_proj, self.model.chns[i], patch_size, patch_size, device="cuda"
                )
                rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                    1
                ).unsqueeze(2).unsqueeze(3)
                self.register_buffer(f"rand_{i}", rand)
        else:
            msg = "Invalid criterion type! Valid models: st or fd"

            raise NotImplementedError(msg)

    def forward_once(self, x, y, z):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = getattr(self, f"rand_{self.index}")
        projx = F.conv2d(x, rand, stride=1)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=1)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)
        projz = F.conv2d(z, rand, stride=1)
        projz = projz.reshape(projz.shape[0], projz.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)
        projz, _ = torch.sort(projz, dim=-1)
        # compute the mean of the sorted convolved input
        return cosine_triplet_loss(projx, projy, projz)

    def fd(self, sr, target, lr):
        # Transform to Fourier Space
        fft_sr = torch.fft.fftn(sr, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))
        fft_lr = torch.fft.fftn(lr, dim=(-2, -1))
        # get the magnitude and phase of the extracted features
        sr_mag = torch.abs(fft_sr)
        sr_phase = torch.angle(fft_sr)

        target_mag = torch.abs(fft_target)
        target_phase = torch.angle(fft_target)

        lr_mag = torch.abs(fft_lr)
        lr_phase = torch.angle(fft_lr)
        s_amplitude = self.forward_once(sr_mag, target_mag, lr_mag)
        s_phase = self.forward_once(sr_phase, target_phase, lr_phase)
        return s_phase + s_amplitude

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, sr, target, lr):
        loss = 0
        b, c, h, w = sr.shape
        if lr.shape != sr.shape:
            lr = F.interpolate(
                lr, size=(h, w), align_corners=True, mode="bicubic"
            ).clamp(0, 1)
        mask = lr==target
        lr[mask] = 1-target[mask]
        sr = self.model(sr)
        target = self.model(target)
        lr = self.model(lr)
        len_perceptual = len(sr)

        for index in range(len_perceptual):
            self.index = index
            loss += self.criterion(
                sr[index],
                target[index],
                lr[index],
            )
        return loss / len_perceptual * self.loss_weight
