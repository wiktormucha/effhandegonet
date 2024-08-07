from models.backbones import BackboneModel
from models.heads import SimpleHead5
import torch
import torch.nn as nn
import torchvision
import numpy as np


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


class ResNet50(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50_backbone = torch.nn.Sequential(
            *(list(resnet50.children())[:-2]))

        self.left_hand = nn.Linear(in_features=(131072), out_features=2)
        self.right_hand = nn.Linear(in_features=(131072), out_features=2)
        self.pooling = nn.MaxPool2d(2)
        self.left_pose = SimpleHead50(in_channels=2048)
        self.right_pose = SimpleHead50(in_channels=2048)

    def forward(self, x):
        features = self.resnet50_backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)
        return self.left_hand(flatten), self.right_hand(flatten), self.left_pose(features), self.right_pose(features)


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


class EffHandEgoNet(nn.Module):
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


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, seq_length, num_heads, dropout, dropout_att):

        super(TransformerEncoder, self).__init__()

        self.MultHeaAtten = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_att)

        self.layer_norm_1 = nn.LayerNorm((seq_length + 1, embed_dim))

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU())

        self.layer_norm_2 = nn.LayerNorm((seq_length + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Encoder Layer

        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        """
        # Calculating self attention
        attn_output, attn_output_weights = self.MultHeaAtten(
            query=x, key=x, value=x, key_padding_mask=None)
        # apply layer normalization on sum of the input and the attention output to get the
        # output of the multi-head attention layer (~1 line)
        out1 = self.layer_norm_1(attn_output + x)
        # pass the output of the multi-head attention layer through a ffn (~1 line)

        ffn_output = self.ffn(out1)

        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        enc_out = self.layer_norm_2(out1 + ffn_output)
        self.dropout(enc_out)

        return enc_out


class ActionTransformer(nn.Module):
    def __init__(self, model_cfg, input_dim=126, out_dim=37, device=0, dataset='h2o') -> None:
        super().__init__()

        input_dim = model_cfg.input_dim
        out_dim = model_cfg.out_dim
        dropout = model_cfg.dropout
        dropout_att = model_cfg.dropout_att
        dropout = model_cfg.dropout
        num_heads = model_cfg.trans_num_heads
        self.num_layers = model_cfg.trans_num_layers
        self.hidden_layers = model_cfg.hidden_layers
        self.seq_length = model_cfg.seq_length
        self.device = device

        # 1) Linear mapper
        self.linear_mapper = nn.Linear(input_dim, self.hidden_layers)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_layers))
        self.encoder_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.encoder_layers.append(TransformerEncoder(
                embed_dim=self.hidden_layers, seq_length=self.seq_length, num_heads=num_heads, dropout=dropout, dropout_att=dropout_att).to(device))

        # 3) CLassification MLP
        if dataset == 'h2o':
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_layers, out_dim),
            )
        elif dataset == 'fpha':
            self.mlp = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.hidden_layers, out_dim),
            )

    def forward(self, x):

        batch_s, _, _ = x.shape
        # 1 Lineralize
        tokens = self.linear_mapper(x)

        # Adding classification token to the tokens
        tokens = torch.stack(
            [torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        tokens += get_positional_embeddings(self.seq_length+1,
                                            self.hidden_layers, device=self.device).repeat(batch_s, 1, 1)

        # Encoder block
        x = tokens

        # Encoder block
        for blk in self.encoder_layers:
            x = blk(x)

        out = x[:, 0]
        out = self.mlp(out)

        return out


def get_positional_embeddings(sequence_lenght, d, device, freq=10000):
    result = torch.ones(sequence_lenght, d)
    for i in range(sequence_lenght):
        for j in range(d):
            result[i][j] = np.sin(i / (freq ** (j / d)) if j %
                                  2 == 0 else np.cos(i / (freq ** ((j - 1) / d))))

    return result.to(device)
