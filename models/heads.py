import sys
import torch.nn as nn
import torch
from models.backbones import ConvBlock
# from models.models import DeconvolutionLayer2
sys.path.append("../")

# from backbones import ConvBlock


class DeconvolutionLayer3(nn.Module):
    """
    Class for deconvolutional layer 
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        """_summary_

        Args:
            in_channels (int): No. of input channels
            out_channels (int):  No. of output channels
            kernel_size (int, optional): Kernel size. Defaults to 2.
            stride (int, optional): Stride size. Defaults to 2.
        """
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """
        out = self.deconv(x)
        return out


class DeconvolutionLayer(nn.Module):
    """
    Class for deconvolutional layer 
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        """_summary_

        Args:
            in_channels (int): No. of input channels
            out_channels (int):  No. of output channels
            kernel_size (int, optional): Kernel size. Defaults to 2.
            stride (int, optional): Stride size. Defaults to 2.
        """
        super().__init__()

        self.deconv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ConvTranspose2d(
                                        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                                    )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """
        out = self.deconv(x)
        return out


class Encoder_Ego(nn.Module):
    """
    Encoder for waterfall architecture
    """

    def __init__(self, in_ch: int) -> None:
        """
        Initialisation

        Args:
            in_ch (int): No. of input channels.
        """
        super().__init__()

        self.deconv1 = DeconvolutionLayer3(in_channels=int(
            in_ch), out_channels=int(in_ch/4), kernel_size=4, stride=4)
        self.deconv2 = DeconvolutionLayer3(in_channels=int(
            in_ch/4), out_channels=int(in_ch/16), kernel_size=4, stride=4)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        out = self.deconv1(x)
        out = self.deconv2(out)
        return out


class Encoder(nn.Module):
    """
    Encoder for waterfall architecture
    """

    def __init__(self, in_ch: int) -> None:
        """
        Initialisation

        Args:
            in_ch (int): No. of input channels.
        """
        super().__init__()

        self.deconv1 = DeconvolutionLayer(in_channels=int(
            in_ch), out_channels=int(in_ch/4), kernel_size=4, stride=4)
        self.deconv2 = DeconvolutionLayer(in_channels=int(
            in_ch/4), out_channels=int(in_ch/16), kernel_size=4, stride=4)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        out = self.deconv1(x)
        out = self.deconv2(out)
        return out

# l1 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=2, dilation=1)
# l2 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=2, dilation=5, padding=0)
# l3 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=2, dilation=8, padding=0)


# l1_1 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=8, dilation=1, padding=0)
# l2_2 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=4, dilation=1, padding=0)
# l2_3 = nn.Conv2d(
#     in_channels=1280, out_channels=1280, kernel_size=4, dilation=1, padding=0)
class WaterfallEgo(nn.Module):
    """
    Waterfall module with atrous convolutions
    """

    def __init__(self) -> None:
        """
        Initialization
        """
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=2, dilation=1),
            # nn.MaxPool2d(2),
            nn.Conv2d(1280, 1280, kernel_size=8, dilation=1)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=2, dilation=5),
            nn.Conv2d(1280, 1280, kernel_size=4, dilation=1),
            # nn.Conv2d(1280, 1280, kernel_size=4, dilation=1)
        )

        self.l3 = nn.Conv2d(1280, 1280, kernel_size=2, dilation=8, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """

        out1 = self.l1(x)
        out2 = self.l2(x)
        out3 = self.l3(x)

        final_out = torch.cat([out1, out2, out3], dim=1)

        return final_out


class Waterfall(nn.Module):
    """
    Waterfall module with atrous convolutions
    """

    def __init__(self) -> None:
        """
        Initialisation
        """
        super().__init__()

        self.l1 = nn.Conv2d(
            in_channels=1280, out_channels=1280, kernel_size=1, dilation=1)
        self.l2 = nn.Conv2d(
            in_channels=1280, out_channels=1280, kernel_size=2, dilation=2)
        self.l3 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        out1 = self.pool2(self.l1(x))
        out2 = self.l2(x)
        out3 = self.l3(x)

        final_out = torch.cat([out1, out2, out3], dim=1)

        return final_out


class SimpleUsampleSingleHand(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.deconv = DeconvolutionLayer(1280, 640)
        self.deconv2 = DeconvolutionLayer(640, 160, kernel_size=4, stride=4)
        self.deconv3 = DeconvolutionLayer(160, 42, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.deconv(x)
        out = self.deconv2(out)
        out = self.deconv3(out)

        return out


class SimpleHead_FPN(nn.Module):
    '''
    This head network works for FPN model 
    '''

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(
            in_channels=in_channels, out_channels=256)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        x = self.deconv1(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class SimpleHead5(nn.Module):
    """
    Simple head containing of 5 transposed layers
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialisation

        Args:
            in_channels (int): No. of input channels.
            out_channels (int): No. of output channels.
        """
        super().__init__()

        self.deconv1 = DeconvolutionLayer(
            in_channels=in_channels, out_channels=256)
        self.deconv2 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv3 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv4 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv5 = DeconvolutionLayer(in_channels=256, out_channels=256)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class SimpleHead_trans(nn.Module):
    """
    Simple head for trans model.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialisation

        Args:
            in_channels (int): No. of input channels.
            out_channels (int): No. of output channels.
        """
        super().__init__()

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=256, kernel_size=2, stride=2,
                                                padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv5 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = self.final(x)
        x = self.sigmoid(x)

        return x


class PredictionHead(nn.Module):
    '''
    Baseline head model generating heatmaps
    '''

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initilisation

        Args:
            input_dim (int): Input dimenssion
            output_dim (int): Output dimenssion
        """
        super().__init__()

        self.convup1 = ConvBlock(input_dim, output_dim)
        self.prob_out = nn.Sigmoid()
        self.upsample128 = nn.Upsample(
            scale_factor=128, mode="bilinear", align_corners=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        x = self.convup1(self.upsample128(x))

        out = self.prob_out(x)
        return out


class PredictionHeadPoints(nn.Module):
    """
    Simple regression head to predict 2D points
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialisation

        Args:
            input_dim (int): Input size
            output_dim (int): No. of points to predict
        """
        super().__init__()

        self.norm = nn.BatchNorm2d(input_dim)
        self.regression = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        self.norm(x)
        x = torch.squeeze(x)

        out = self.regression(x)
        return out
