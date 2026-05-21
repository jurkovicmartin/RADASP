import torch
import torch.nn as nn
import torchxrayvision as xrv


class XRVEncoder(nn.Module):
    """
    A wrapper for torchxrayvision models to serve as an Encoder (feature extractor / backbone).
    """
    def __init__(self, weight_name: str ="densenet121-res224-all", target_filters: list[int] =[32, 64, 128, 256, 512], input_channels: int =1):
        """
        Initializes the torchxrayvision wrapper.

        Args:
            weight_name (str): The specific XRV weights to load.
            target_filters (list[int], optional): The target number of filters for each stage. Default is [32, 64, 128, 256, 512].
            input_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()
        self.weight_name = weight_name
        self.target_filters = target_filters
        
        # STAGE 0: The "High-Resolution" Stem (1/1 Res)
        self.stage0 = nn.Sequential(
            nn.Conv2d(input_channels, target_filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(target_filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # STAGES 1-4: The XRV Backbone (1/2 to 1/16 Res)
        if "densenet" in self.weight_name:
            self._build_densenet_stages()
            self.backbone_channels = [64, 256, 512, 1024]
        elif "resnet" in self.weight_name:
            self._build_resnet_stages()
            self.backbone_channels = [64, 256, 512, 1024]
        else:
            raise ValueError("Unsupported backbone")

        # 1x1 convolutions to compress the backbone features
        self.projectors = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                                         for in_ch, out_ch in zip(self.backbone_channels, self.target_filters[1:])])


    def _build_densenet_stages(self):
        base_model = xrv.models.DenseNet(weights=self.weight_name)
        features = base_model.features
        self.stage1 = nn.Sequential(features.conv0, features.norm0, features.relu0) # 1/2 res
        self.stage2 = nn.Sequential(features.pool0, features.denseblock1)           # 1/4 res
        self.stage3 = nn.Sequential(features.transition1, features.denseblock2)     # 1/8 res
        self.stage4 = nn.Sequential(features.transition2, features.denseblock3)     # 1/16 res


    def _build_resnet_stages(self):
        base_model = xrv.models.ResNet(weights=self.weight_name)
        resnet = base_model.model if hasattr(base_model, 'model') else base_model
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 1/2 res
        self.stage2 = nn.Sequential(resnet.maxpool, resnet.layer1)         # 1/4 res
        self.stage3 = resnet.layer2                                        # 1/8 res
        self.stage4 = resnet.layer3                                        # 1/16 res


    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns:
            list[torch.Tensor]: 5 feature maps at [1/1, 1/2, 1/4, 1/8, 1/16] resolutions.
        """
        f0 = self.stage0(input)
        f1 = self.stage1(input)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        features = [f1, f2, f3, f4]

        return [f0] + [proj(feature) for proj, feature in zip(self.projectors, features)]
    



### ENCODER TEST

def create_dummy_batch(batch_size: int, channels: int, img_size: int) -> torch.Tensor:
    return torch.randn(batch_size, channels, img_size, img_size)


def test_xrv_encoder(model: str, batch_size: int =2, img_size: int =256):
    """
    Tests the XRVEncoder with a dummy batch to verify output shapes and spatial dimensions.

    Args:
        model (str): The name of the XRV weight to test.
        batch_size (int, optional): Number of images in the dummy batch. Defaults to 2.
        img_size (int, optional): Spatial size of the dummy images. Defaults to 256.
    """
    channels = 1
    # Expected channel depths
    expected_channels = [32, 64, 128, 256, 512]

    dummy_batch = create_dummy_batch(batch_size, channels, img_size)

    print(f"=== Testing Model: {model} ===")
    try:
        encoder = XRVEncoder(model, expected_channels)
        encoder.eval()
        
        with torch.no_grad():
            features = encoder(dummy_batch)
        
        # Verify the number of outputs
        assert len(features) == 5, f"Error: Expected 5 feature maps, got {len(features)}"
        print("Success! Extracted 5 projected feature maps. Here are their shapes:")
        for i, f in enumerate(features):
            expected_size = img_size // (2 ** i)
            expected_ch = expected_channels[i]
            
            print(f"  Stage {i} (1/{2**i} res): \tShape = {list(f.shape)} "
                  f"  -> Channels: {f.shape[1]} (Expected: {expected_ch}), "
                  f"Spatial: {f.shape[2]}x{f.shape[3]}")
            
            # Check the spatial dimensions
            assert f.shape[2] == expected_size and f.shape[3] == expected_size, \
                f"Spatial dimension mismatch at stage {i}! Expected {expected_size}x{expected_size}, got {f.shape[2]}x{f.shape[3]}"
            
            # Check the projected channel dimensions
            assert f.shape[1] == expected_ch, \
                f"Channel dimension mismatch at stage {i}! Expected {expected_ch}, got {f.shape[1]}"
        print("-" * 50 + "\n")   
    except Exception as e:
        print(f"Test failed for {model}! Error: {e}\n")
        

if __name__ == "__main__":
    resnet = "resnet50-res512-all"
    densenet = "densenet121-res224-all"
    # densenet = "densenet121-res224-nih"

    test_xrv_encoder(resnet, img_size=512)
    test_xrv_encoder(densenet, img_size=224)