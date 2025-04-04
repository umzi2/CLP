import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg


class VGG(nn.Module):
    def __init__(self, requires_grad=False, vgg_weights=None):
        super().__init__()
        vgg_pretrained_features = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        vgg_pretrained_features.eval()

        if vgg_weights is None:
            self.vgg_weights = (0.5, 0.5, 1.0, 1.0, 1.0)
        else:
            self.vgg_weights = vgg_weights

        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()

        # vgg19
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 128, 256, 512, 512]

    def get_features(self, x):
        # normalize the data
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h * self.vgg_weights[0]

        h = self.stage2(h)
        h_relu2_2 = h * self.vgg_weights[1]

        h = self.stage3(h)
        h_relu3_3 = h * self.vgg_weights[2]

        h = self.stage4(h)
        h_relu4_3 = h * self.vgg_weights[3]

        h = self.stage5(h)
        h_relu5_3 = h * self.vgg_weights[4]

        # get the features of each layer
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x):
        return self.get_features(x)


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


def generate_uniform_noise(shape, min_val=0.0, max_val=1.0, device="cpu"):
    return torch.empty(shape).uniform_(min_val, max_val)


class CLPLoss(nn.Module):
    def __init__(
        self, model="dinov2", loss_weight: float = 1.0, flatten: int = 2
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        model = model.lower()
        if model == "vgg":
            self.model = VGG()
        elif model == "dinov2":
            self.model = dinov2()
        else:
            msg = "Invalid model type! Valid models: VGG or DINOv2"
            raise NotImplementedError(msg)
        self.flatten = flatten
        self.criterion = nn.TripletMarginLoss()

    def forward(self, sr, target, lr):
        loss = 0
        b, c, h, w = sr.shape
        if lr.shape != sr.shape:
            lr = F.interpolate(
                lr, size=(h, w), align_corners=True, mode="bicubic"
            ).clamp(0, 1)
        mask = lr == target
        lr[mask] = 1 - target[mask]
        sr = self.model(sr)
        target = self.model(target)
        lr = self.model(lr)
        len_perceptual = len(sr)

        for index in range(len_perceptual):
            loss += self.criterion(
                sr[index].flatten(start_dim=self.flatten),
                target[index].flatten(start_dim=self.flatten),
                lr[index].flatten(start_dim=self.flatten),
            )
        return loss / len_perceptual * self.loss_weight
