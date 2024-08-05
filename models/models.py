from models.backbones import BackboneModel, BackboneModel
from models.heads import SimpleHead5
import torch
import torch.nn as nn
import torchvision


class DeconvolutionLayer2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2, padding=0, last=False) -> None:

        super().__init__()

        self.last = last

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:

        out = self.deconv(x)

        if not self.last:
            out = self.norm(out)

        return out


class SimpleHead50(nn.Module):
    def __init__(self, in_channels=1280, out_channels=21) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer2(
            in_channels=in_channels, out_channels=256)
        self.deconv2 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv3 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv4 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv5 = DeconvolutionLayer2(
            in_channels=256, out_channels=256, last=True)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv5(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class HandUpSampler(nn.Module):
    def __init__(self, inpt_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sampler = nn.Sequential(
            DeconvolutionLayer2(
                inpt_channels, 256, kernel_size=4, stride=2, padding=1, last=False),
            DeconvolutionLayer2(
                256, 256, kernel_size=4, stride=2, padding=1, last=False),
            DeconvolutionLayer2(
                256, 256, kernel_size=4, stride=2, padding=1, last=True),
            nn.Conv2d(256, 21, kernel_size=1, stride=1)
        )

    def forward(self, x):

        return self.sampler(x)


class ConvNext3Egocentric(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        convnext = torchvision.models.convnext_tiny(
            weights='DEFAULT', progress=True)
        self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-2]))

        self.left_hand = nn.Linear(in_features=(49152), out_features=2)
        self.right_hand = nn.Linear(in_features=(49152), out_features=2)
        self.pooling = nn.MaxPool2d(2)

        self.left_pose = SimpleHead50(in_channels=768)
        self.right_pose = SimpleHead50(in_channels=768)

    def forward(self, x):

        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        return self.left_hand(flatten), self.right_hand(flatten), self.left_pose(features), self.right_pose(features)


class MobileNetV3Egocentric(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        mobilenetv3 = torchvision.models.mobilenet_v3_large(
            weights='DEFAULT', progress=True)
        self.backbone = torch.nn.Sequential(
            *(list(mobilenetv3.children())[:-2]))

        self.left_hand = nn.Linear(in_features=(61440), out_features=2)
        self.right_hand = nn.Linear(in_features=(61440), out_features=2)
        self.pooling = nn.MaxPool2d(2)

        self.left_pose = SimpleHead50(in_channels=960)
        self.right_pose = SimpleHead50(in_channels=960)

    def forward(self, x):

        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        return self.left_hand(flatten), self.right_hand(flatten), self.left_pose(features), self.right_pose(features)


class SwinV2Egocentric(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        swinv2 = torchvision.models.swin_v2_t(weights='DEFAULT', progress=True)
        self.backbone = torch.nn.Sequential(*(list(swinv2.children())[:-3]))

        self.left_hand = nn.Linear(in_features=(49152), out_features=2)
        self.right_hand = nn.Linear(in_features=(49152), out_features=2)
        self.pooling = nn.MaxPool2d(2)

        self.left_pose = SimpleHead50(in_channels=768)
        self.right_pose = SimpleHead50(in_channels=768)

    def forward(self, x):

        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        return self.left_hand(flatten), self.right_hand(flatten), self.left_pose(features), self.right_pose(features)


# class NewModel(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         resnet50 = torch.hub.load(
#             'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#         self.resnet50_backbone = torch.nn.Sequential(
#             *(list(resnet50.children())[:-2]))

#         # self.left = HandUpSampler(inpt_channels=2048)
#         # self.right = HandUpSampler(inpt_channels=2048)
#         # self.left_prob = nn.Sigmoid()
#         # self.right_prob = nn.Sigmoid()

#         self.left_hand = nn.Linear(in_features=(131072), out_features=2)
#         self.right_hand = nn.Linear(in_features=(131072), out_features=2)
#         self.pooling = nn.MaxPool2d(2)

#         # self.left_pose = nn.Sequential(
#         #     HandUpSampler(inpt_channels=1280),
#         #     nn.Sigmoid()
#         # )

#         self.left_pose = SimpleHead50(in_channels=2048)

#         # self.right_pose = nn.Sequential(
#         #     HandUpSampler(inpt_channels=1280),
#         #     nn.Sigmoid()
#         # )

#         self.right_pose = SimpleHead50(in_channels=2048)

#     def forward(self, x):
#         features = self.resnet50_backbone(x)
#         # print(features.shape)
#         # left = self.left(features)
#         # right = self.right(features)

#         flatten = torch.flatten(self.pooling(features), 1)
#         # print(flatten.shape)
#         # left_hand = self.left_hand(flatten)
#         # right_hand = self.right_hand(flatten)

#         # return None, None, None, features
#         return self.left_hand(flatten), self.right_hand(flatten), self.left_pose(features), self.right_pose(features)
#         # return self.left_prob(left), self.right_prob(right)


class EffHandEgoNet_FPHAB(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = BackboneModel()
        self.pooling = nn.MaxPool2d(2)
        self.hand_pose = SimpleHead50()
        self.obj_class = nn.Linear(in_features=81920, out_features=27)

    def forward(self, x):

        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        return self.obj_class(flatten), self.hand_pose(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_hand_pose(self):
        for param in self.hand_pose.parameters():
            param.requires_grad = False


class CustomEgocentric(nn.Module):
    def __init__(self, handness_in: int = 81920, handness_out: int = 2, *args, **kwargs) -> None:
        """Initilise the model
        Args:
            handness_in (int, optional): _description_. Defaults to 81920 for 512 input image.
            handness_out (int, optional): _description_. Defaults to 2, binary classification.
        """

        super().__init__(*args, **kwargs)

        self.backbone = BackboneModel()

        self.left_hand = nn.Linear(
            in_features=handness_in, out_features=handness_out)
        self.right_hand = nn.Linear(
            in_features=handness_in, out_features=handness_out)
        self.pooling = nn.MaxPool2d(2)

        self.left_pose = SimpleHead50()
        self.right_pose = SimpleHead50()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): input tensor shape B,3,512,512

        Returns:
            torch.Tensor: handness_lef, handness_right, left pose, right pose
        """
        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        return {
            'left_handness': self.left_hand(flatten),
            'right_handness': self.right_hand(flatten),
            'left_2D_pose': self.left_pose(features),
            'right_2D_pose': self.right_pose(features),
        }


# class CustomEgocentricSingle(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.backbone = BackboneModel()

#         # self.left_hand = nn.Linear(in_features=81920, out_features=2)
#         # self.right_hand = nn.Linear(in_features=81920, out_features=2)
#         # self.pooling = nn.MaxPool2d(2)

#         # self.left_pose = nn.Sequential(
#         #     HandUpSampler(inpt_channels=1280),
#         #     nn.Sigmoid()
#         # )

#         # self.left_pose = SimpleHead50()

#         # self.right_pose = nn.Sequential(
#         #     HandUpSampler(inpt_channels=1280),
#         #     nn.Sigmoid()
#         # )

#         self.right_pose = SimpleHead50()

#     def forward(self, x):
#         features = self.backbone(x)
#         # left = self.left(features)
#         # right = self.right(features)

#         # flatten = torch.flatten(self.pooling(features), 1)
#         # left_hand = self.left_hand(flatten)
#         # right_hand = self.right_hand(flatten)

#         # return None, None, None, features
#         # , self.left_pose(features),
#         return self.right_pose(features)


class EffHandNet(nn.Module):
    """
    EffHandWSimple model architecture
    """

    def __init__(self, in_channel: int = 1280, out_channel: int = 21) -> None:
        """
        Initialisation

        Args:
            in_channel (int, optional): No. of input channels. Defaults to 1280.
            out_channel (int, optional): No. of output channels. Defaults to 21.
        """
        super().__init__()

        self.backbone = BackboneModel()
        self.head = SimpleHead5(in_channel, out_channel)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        bck = self.backbone(x)
        out = self.head(bck)

        return out
