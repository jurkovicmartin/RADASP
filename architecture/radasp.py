import torch
import torch.nn as nn

from architecture.modules import VGG, ASPP, CBAM
from architecture.encoder import XRVEncoder


class RADASP(nn.Module):
    """
    Residual Attention Dense Atrous Spatial Pyramid (RADASP).
    """
    def __init__(self, num_classes: int =1, input_channels: int =1, deep_supervision: bool =False):
        """
        Initializes RADASP model.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1.
            input_channels (int, optional): Number of input channels. Defaults to 1.
            deep_supervision (bool, optional): If True, the model will output a prediction at each resolution level. Defaults to False.
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        # Number of filters
        NB_FILTER = [32, 64, 128, 256, 512]
        backbone = "densenet121-res224-all"

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.encoder = XRVEncoder(backbone, NB_FILTER, input_channels)

        self.decoder = nn.ModuleDict({
            "aspp": ASPP(NB_FILTER[3], NB_FILTER[3], [4, 8, 12]),

            "conv0_1": CBAM(NB_FILTER[0]+NB_FILTER[1], NB_FILTER[0]),
            "conv1_1": CBAM(NB_FILTER[1]+NB_FILTER[2], NB_FILTER[1]),
            "conv2_1": CBAM(NB_FILTER[2]+NB_FILTER[3], NB_FILTER[2]),

            "conv0_2": CBAM(NB_FILTER[0]*2+NB_FILTER[1], NB_FILTER[0]),
            "conv1_2": CBAM(NB_FILTER[1]*2+NB_FILTER[2], NB_FILTER[1]),

            "conv0_3": CBAM(NB_FILTER[0]*3+NB_FILTER[1], NB_FILTER[0])
        })

        # Output
        if self.deep_supervision:
            self.decoder["final1"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
            self.decoder["final2"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
            self.decoder["final3"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
        else:
            self.decoder["final"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)


    def forward(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The output tensor(s) of the model -- depending on deep_supervision.
        """
        x0_0, x1_0, x2_0, x3_0, _ = self.encoder(input)

        # Applying ASPP
        x3_0 = self.decoder["aspp"](x3_0)

        x0_1 = self.decoder["conv0_1"](torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.decoder["conv1_1"](torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.decoder["conv0_2"](torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.decoder["conv2_1"](torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.decoder["conv1_2"](torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.decoder["conv0_3"](torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if self.deep_supervision:
            out1 = self.decoder["final1"](x0_1)
            out2 = self.decoder["final2"](x0_2)
            out3 = self.decoder["final3"](x0_3)
            return [out1, out2, out3]
        else:
            return self.decoder["final"](x0_3)
        

class DASP(nn.Module):
    """
    Dense Atrous Spatial Pyramid (DASP) -- RADASP without attention modules.
    """
    def __init__(self, num_classes: int =1, input_channels: int =1, deep_supervision: bool =False):
        """
        Initializes DASP model.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1.
            input_channels (int, optional): Number of input channels. Defaults to 1.
            deep_supervision (bool, optional): If True, the model will output a prediction at each resolution level. Defaults to False.
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        # Number of filters
        NB_FILTER = [32, 64, 128, 256, 512]
        backbone = "densenet121-res224-all"

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.encoder = XRVEncoder(backbone, NB_FILTER, input_channels)

        self.decoder = nn.ModuleDict({
            "aspp": ASPP(NB_FILTER[3], NB_FILTER[3], [4, 8, 12]),

            "conv0_1": VGG(NB_FILTER[0]+NB_FILTER[1], NB_FILTER[0], NB_FILTER[0]),
            "conv1_1": VGG(NB_FILTER[1]+NB_FILTER[2], NB_FILTER[1], NB_FILTER[1]),
            "conv2_1": VGG(NB_FILTER[2]+NB_FILTER[3], NB_FILTER[2], NB_FILTER[2]),

            "conv0_2": VGG(NB_FILTER[0]*2+NB_FILTER[1], NB_FILTER[0], NB_FILTER[0]),
            "conv1_2": VGG(NB_FILTER[1]*2+NB_FILTER[2], NB_FILTER[1], NB_FILTER[1]),

            "conv0_3": VGG(NB_FILTER[0]*3+NB_FILTER[1], NB_FILTER[0], NB_FILTER[0])
        })

        if self.deep_supervision:
            self.decoder["final1"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
            self.decoder["final2"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
            self.decoder["final3"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)
        else:
            self.decoder["final"] = nn.Conv2d(NB_FILTER[0], num_classes, kernel_size=1)


    def forward(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The output tensor(s) of the model -- depending on deep_supervision.
        """
        x0_0, x1_0, x2_0, x3_0, _ = self.encoder(input)

        # Applying ASPP
        x3_0 = self.decoder["aspp"](x3_0)

        x0_1 = self.decoder["conv0_1"](torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.decoder["conv1_1"](torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.decoder["conv0_2"](torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.decoder["conv2_1"](torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.decoder["conv1_2"](torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.decoder["conv0_3"](torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if self.deep_supervision:
            out1 = self.decoder["final1"](x0_1)
            out2 = self.decoder["final2"](x0_2)
            out3 = self.decoder["final3"](x0_3)
            return [out1, out2, out3]
        else:
            return self.decoder["final"](x0_3)    

