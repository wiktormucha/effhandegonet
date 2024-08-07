import torch.nn as nn
import torch


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
