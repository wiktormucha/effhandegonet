import torch
import torch.nn as nn
import torchvision
from torchvision.models import EfficientNet_V2_S_Weights
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_depth: int, out_depth: int) -> None:
        """
        Convolution block with batch normalisation, conv btch normalisation, conv

        Args:
            in_depth (int): Input no. of channels
            out_depth (int): Output no. of channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """
        return self.double_conv(x)


class EfficientNetV2(nn.Module):
    """
    Efficientnet backbone
    """

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(
            *(list(self.backbone.children())[:1]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """
        return self.backbone(x)


class BackboneModel_FPN(nn.Module):
    '''
    This backbone stacks feautures from different level of extractor
    '''

    def __init__(self):
        super().__init__()

        self.model = EfficientNetV2()

        self.seq0 = self.model.backbone[0][0]
        self.seq1 = self.model.backbone[0][1]
        self.seq2 = self.model.backbone[0][2]
        self.seq3 = self.model.backbone[0][3]
        self.seq4 = self.model.backbone[0][4]
        self.seq5 = self.model.backbone[0][5]
        self.seq6 = self.model.backbone[0][6]
        self.seq7 = self.model.backbone[0][7]

        self.upsamle = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        out0 = self.seq0(x)
        out1 = self.seq1(out0)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        out4 = self.seq4(out3)
        out5 = self.seq5(out4)
        out6 = self.seq6(out5)
        out7 = self.seq7(out6)

        merge67 = torch.cat([out6, out7], dim=1)
        merge67 = self.upsamle(merge67)
        merge45 = torch.cat([out4, out5], dim=1)
        merge4567 = torch.cat([merge45, merge67], dim=1)
        merge4567 = self.upsamle(merge4567)
        merge34567 = torch.cat([out3, merge4567], dim=1)
        merge34567 = self.upsamle(merge34567)
        merge234567 = torch.cat([out2, merge34567], dim=1)
        merge234567 = self.upsamle(merge234567)

        final = merge01234567 = torch.cat([out0, out1, merge234567], dim=1)

        return final


class BackboneModel_Depth(nn.Module):
    """
    Efficientnet backbone without last max pooling. Outputs dimmensions of Bx4x4x1280.
    """

    def __init__(self, num_input_channels: int = 4):
        """
        Init
        """
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(
            *(list(self.backbone.children())[:-2]))

        # self.backbone

        self.backbone[0][0][0] = torch.nn.Conv2d(
            num_input_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        return self.backbone(x)


class BackboneModel(nn.Module):
    """
    Efficientnet backbone without last max pooling. Outputs dimmensions of Bx4x4x1280.
    """

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(
            *(list(self.backbone.children())[:-2]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        return self.backbone(x)


class HighLowFeauturesBck(nn.Module):
    """
    This backbone extracts high and low level feautrues from the model.

    """

    def __collect(self, m, i, o):

        self.act["exp_after_linear"] = o.detach()
        return self.act["exp_after_linear"]

    def __init__(self) -> None:
        """
        Model initialisation
        """
        super().__init__()

        self.model = EfficientNetV2()
        self.model.backbone[0][2].register_forward_hook(self.__collect)
        self.act = {}

    def forward(self, x: torch.tensor) -> dict:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            dict:   'low_feautures' - Low level feautres after 2nd module in backbone
                    'high_feautures' - outputs form final layer
        """

        out_high = self.model(x)
        out_low = self.act["exp_after_linear"]

        return {
            'low_feautures': out_low,
            'high_feautures': out_high
        }
