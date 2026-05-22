import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A ResNet-style block that allows sharp aerodynamic gradients to bypass 
    convolutional blurring via an identity skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class   UNetPhysicsSurrogate(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. The Encoder (Downsampling)
        for feature in features:
            self.downs.append(ResidualBlock(in_channels, feature))
            in_channels = feature
        
        # 2. The Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2) # 512 -> 1024

        # 3. Auxiliary Head
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(features[-1] * 2, 256), # 1024 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) # Output layer: 2 neurons for [Cl, Cd]
        )

        # 4. The Decoder 
        for feature in reversed(features):
            # Transposed Conv scales up spatial dims by 2, halves the channels
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(feature*2, feature, kernel_size=3, padding=1, padding_mode='replicate')
                )
            )
            # After concatenation with skip connection, channels double, so we reduce
            self.ups.append(ResidualBlock(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        # Pass through Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Save for the skip connection
            x = self.pool(x)
        
        # Pass through Bottleneck
        x = self.bottleneck(x)

        # Branch the latent representation into the auxiliary head
        coeffs = self.aux_head(x)

        # Reverse the skip connections list to match the decoder's order
        skip_connections = skip_connections[::-1] 

        # Pass through Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Upsample
            skip_connection = skip_connections[i//2]
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_x) 

        pressure_field = self.final_conv(x)
        
        return pressure_field, coeffs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Phase 4 ResNet Architecture on: {device}")
    model = UNetPhysicsSurrogate().to(device)
    dummy = torch.randn(8, 2, 128, 128).to(device)
    p_out, c_out = model(dummy)
    print("Success! Tensor shapes match. Ready for training.")