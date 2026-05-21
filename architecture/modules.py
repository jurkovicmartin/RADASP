import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) block.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: list[int] =[6, 12, 18]):
        """
        Initializes an ASPP block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation_rates (list[int], optional): Dilation rates for the atrous convolutions. Defaults to [6, 12, 18].
        """
        super().__init__()
        
        # The 1x1 Convolution branch (Captures the original scale)
        self.branch_1x1 = ASPPConvBranch(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        
        # The Dilated Convolution branches (Captures multiple larger scales)
        self.branch_dilated_1 = ASPPConvBranch(in_channels, out_channels, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0])
        self.branch_dilated_2 = ASPPConvBranch(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1])
        self.branch_dilated_3 = ASPPConvBranch(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2])
        
        # The Global Average Pooling branch (Captures the whole-image context)
        self.branch_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # The Final Projector
        self.projector = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res_1x1 = self.branch_1x1(input)
        res_d1 = self.branch_dilated_1(input)
        res_d2 = self.branch_dilated_2(input)
        res_d3 = self.branch_dilated_3(input)
        
        res_global = self.branch_global_pool(input)
        # Upsample the result back to the spatial size of the input
        res_global = F.interpolate(res_global, size=input.shape[2:], mode="bilinear", align_corners=False)
        
        # Concatenate all 5 feature maps along the channel dimension
        concatenated_features = torch.cat([res_1x1, res_d1, res_d2, res_d3, res_global], dim=1)
        
        # Project the massive concatenated tensor back down to out_channels
        final_output = self.projector(concatenated_features)
        
        return final_output
    

class ASPPConvBranch(nn.Module):
    """
    A helper module for the standard and dilated convolution branches.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int):
        """
        Initializes an ASPPConvBranch module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            padding (int): Convolution padding.
            dilation (int): Convolution dilation rate.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) block.
    """
    def __init__(self, in_channels: int, out_channels: int, reduce_ratio: int =8, spatial_kernel_size: int =7, pure: bool =False):
        """
        Initializes a CBAM block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            reduce_ratio (int, optional): Reduction ratio for the CAM bottleneck. Defaults to 8.
            spatial_kernel_size (int, optional): Kernel size for the SAM convolution. Defaults to 7.
            pure (bool, optional): If True, skips the initial 3x3 convolution and residual connection. Defaults to False.
        """
        super().__init__()
        self.pure = pure

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cam = CAM(out_channels, reduce_ratio)
        self.sam = SAM(spatial_kernel_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pure:
            att = self.cam(x)
            att = self.sam(att)
            return att
        else:
            x = self.conv(x)
            att = self.cam(x)
            att = self.sam(att)

            return x + att
        

class CAM(nn.Module):
    """
    Channel Attention Module (CAM) block.
    """
    def __init__(self, channels: int, reduce_ratio: int =8):
        """
        Initializes a CAM block.

        Args:
            channels (int): Number of input channels.
            reduce_ratio (int, optional): Reduction ratio for the MLP bottleneck. Defaults to 8.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduce_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduce_ratio, channels, 1, bias=False)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = self.mlp(F.adaptive_max_pool2d(x, output_size=1))
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, output_size=1))
        out = avg_out + max_out

        return x * torch.sigmoid(out)
    

class SAM(nn.Module):
    """
    Spatial Attention Module (SAM) block.
    """
    def __init__(self, kernel_size: int =7):
        """
        Initializes a SAM block.

        Args:
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)

        return x * torch.sigmoid(out)


class VGG(nn.Module):
    """
    Visual Geometry Group (VGG) block.
    """
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        """
        Initializes a VGGBlock.

        Args:
            in_channels (int): Number of input channels.
            middle_channels (int): Number of middle channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out