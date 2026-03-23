"""
TODO: keep all model functions and so on in one place look for any other stragglers
My saviours for this was:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
https://github.com/microsoft/Swin-Transformer
Official documentation and PyTorch forums (Grayscale ResNet weight averaging technique)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Tuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#SwinDAMFN
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class MultiSeparableAttention(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        mid = out_ch
        self.b3 = DepthwiseSeparableConv(in_ch, mid, k=3, p=1)
        self.b5 = DepthwiseSeparableConv(in_ch, mid, k=5, p=2)
        self.b7 = DepthwiseSeparableConv(in_ch, mid, k=7, p=3)

        self.fc1 = nn.Conv2d(mid*3, (mid*3)//reduction, 1)
        self.fc2 = nn.Conv2d((mid*3)//reduction, mid*3, 1)

        self.out = nn.Sequential(
            nn.Conv2d(mid*3, out_ch, 1),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_cat = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        w = F.adaptive_avg_pool2d(x_cat, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return self.out(x_cat * w)


class TriShuffleConvAttention(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(out_ch*3, (out_ch*3)//reduction, 1)
        self.fc2 = nn.Conv2d((out_ch*3)//reduction, out_ch*3, 1)
        self.out = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.flip(self.conv(torch.flip(x, dims=[-1])), dims=[-1]) 
        y3 = torch.flip(self.conv(torch.flip(x, dims=[-2])), dims=[-2])
        y_cat = torch.cat([y1, y2, y3], dim=1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(y_cat, 1)))))
        return self.out(y_cat * w)


class DAMFNBranch(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.msa1 = MultiSeparableAttention(base, base*2)
        self.pool1 = nn.MaxPool2d(2)
        self.tsca1 = TriShuffleConvAttention(base*2, base*4)
        self.pool2 = nn.MaxPool2d(2)
        self.msa2 = MultiSeparableAttention(base*4, base*8)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.msa1(x))
        x = self.pool2(self.tsca1(x))
        return self.msa2(x)


class TripletAttentionFusion(nn.Module):
    def __init__(self, swin_ch, cnn_ch, out_ch):
        super().__init__()
        self.swin_proj = nn.Conv2d(swin_ch, out_ch, 1, bias=False)
        self.cnn_proj = nn.Conv2d(cnn_ch, out_ch, 1, bias=False)

        self.h = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.w = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.c = nn.Conv2d(out_ch, out_ch, 1, bias=False)

        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)   
        )

    def forward(self, s, c):
        if s.shape[-1] != c.shape[-1]:
            s = F.interpolate(s, size=c.shape[-2:], mode="bilinear", align_corners=False)

        x = self.swin_proj(s) + self.cnn_proj(c)

        h = torch.sigmoid(self.h(torch.mean(x, dim=3, keepdim=True)))
        w = torch.sigmoid(self.w(torch.mean(x, dim=2, keepdim=True)))
        c = torch.sigmoid(self.c(F.adaptive_avg_pool2d(x, 1)))

        x = (x*h + x*w + x*c) / 3 
        return self.out(x)


class SwinDAMFN(nn.Module):
    def __init__(self, num_classes=2, image_size=224):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(3,),
            in_chans=3,
            img_size=image_size,
            strict_img_size=False
        )
        swin_ch = self.swin.feature_info[-1]["num_chs"]

        self.damfn = DAMFNBranch(in_ch=1, base=32)
        cnn_ch = 32 * 8

        self.fusion = TripletAttentionFusion(swin_ch, cnn_ch, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Swin branch (repeat grayscale to 3 channels)
        s = self.swin(x.repeat(1, 3, 1, 1)) 
        if isinstance(s, (list, tuple)):
            s = s[-1]
        if s.dim() == 4 and s.shape[-1] in [96, 192, 384, 768, 1024]:
            s = s.permute(0, 3, 1, 2).contiguous()

        # CNN branch
        c = self.damfn(x)

        f = self.fusion(s, c)
        return self.fc(self.gap(f).flatten(1))


#ResNet

def build_resnet18_grayscale(num_classes: int = 2, dropout_p: float = 0.5) -> nn.Module:
    """
    https://discuss.pytorch.org/t/need-help-can-i-modify-a-batchnorm2ds-num-features/18685/8
    and other posts were helpful for these model setups
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    in_features = model.fc.in_features  
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(128, num_classes),    
    )
    return model


def build_resnet50_grayscale(num_classes: int = 2, dropout_p: float = 0.6) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(256, num_classes),
    )
    return model


# Helpers Grad-CAM etc

def build_model(model_name: str) -> nn.Module:
    """Unified factory — exactly what your Gradio app needs."""
    if model_name == "ResNet18":
        return build_resnet18_grayscale(num_classes=2, dropout_p=0.5)
    if model_name == "ResNet50":
        return build_resnet50_grayscale(num_classes=2, dropout_p=0.6)
    if model_name == "SwinDAMFN":
        return SwinDAMFN(num_classes=2)
    raise ValueError(f"Unsupported model: {model_name}")


def get_target_layer(model: nn.Module):
    """
    Get the last conv layer of model
    """
    #Check which nmodel
    if isinstance(model, models.ResNet):
        block = model.layer4[-1]
        return block.conv3 if hasattr(block, "conv3") else block.conv2
    if isinstance(model, SwinDAMFN):
        return model.fusion.out[0]          # fusion output conv
    raise ValueError("Could not determine target layer for this model.")


def load_model(model_label: str, checkpoint_path: str) -> nn.Module:
    """
    For the web app to load model
    """
    if "ResNet18" in model_label:
        model = build_model("ResNet18")
    elif "ResNet50" in model_label:
        model = build_model("ResNet50")
    else:
        model = build_model("SwinDAMFN")

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model
