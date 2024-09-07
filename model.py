import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConcentricRingDetection(nn.Module):
    def __init__(self, in_channels, out_channels, num_rings=5):
        super(ConcentricRingDetection, self).__init__()
        self.num_rings = num_rings
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        features = self.conv(x)
        b, c, h, w = features.size()
        center = torch.tensor([h/2, w/2]).to(features.device)
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float, device=features.device),
                              torch.arange(w, dtype=torch.float, device=features.device),
                              indexing='ij')
        dist = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        rings = []
        for i in range(self.num_rings):
            ring = torch.exp(-(dist - (i+1)*h/self.num_rings/2)**2 / (h/self.num_rings/2)**2)
            rings.append(features * ring.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(rings, dim=1)

class SpiralArmDetection(nn.Module):
    def __init__(self, in_channels, out_channels, num_arms=4):
        super(SpiralArmDetection, self).__init__()
        self.num_arms = num_arms
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        features = self.conv(x)
        b, c, h, w = features.size()
        center = torch.tensor([h/2, w/2]).to(features.device)
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float, device=features.device),
                              torch.arange(w, dtype=torch.float, device=features.device),
                              indexing='ij')
        angle = torch.atan2(y - center[0], x - center[1])
        dist = torch.log(torch.sqrt((x - center[1])**2 + (y - center[0])**2) + 1e-5)
        
        arms = []
        for i in range(self.num_arms):
            arm = torch.sin(angle - 2*np.pi*i/self.num_arms - dist)
            arms.append(features * arm.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(arms, dim=1)

class IntensityGradientLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IntensityGradientLayer, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        features = self.conv(x)
        b, c, h, w = features.size()
        center = torch.tensor([h/2, w/2]).to(features.device)
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float, device=features.device),
                              torch.arange(w, dtype=torch.float, device=features.device),
                              indexing='ij')
        dist = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        max_dist = torch.sqrt(torch.tensor(h**2 + w**2)).to(features.device) / 2
        gradient = 1 - (dist / max_dist)
        return features * gradient.unsqueeze(0).unsqueeze(0)

class AsymmetryDetectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsymmetryDetectionLayer, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        features = self.conv(x)
        b, c, h, w = features.size()
        center = torch.tensor([h/2, w/2]).to(features.device)
        q1 = features[:, :, :int(center[0]), :int(center[1])]
        q2 = features[:, :, :int(center[0]), int(center[1]):]
        q3 = features[:, :, int(center[0]):, :int(center[1])]
        q4 = features[:, :, int(center[0]):, int(center[1]):]
        asymmetry = torch.abs(q1.mean(dim=(2,3)) - q3.mean(dim=(2,3))) + torch.abs(q2.mean(dim=(2,3)) - q4.mean(dim=(2,3)))
        asymmetry = asymmetry.sum(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
        return features * asymmetry

class AdvancedCycloneModel(nn.Module):
    def __init__(self):
        super(AdvancedCycloneModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cpu")

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConcentricRingDetection(64, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            SpiralArmDetection(32 * 5, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            IntensityGradientLayer(64 * 4, 128),
            
            AsymmetryDetectionLayer(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def to(self, device):
        self.device = device
        return super(AdvancedCycloneModel, self).to(device)

    def forward(self, x):
        self.logger.info(f"Input shape: {x.shape}")
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            self.logger.info(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        x = torch.flatten(x, 1)
        self.logger.info(f"After flatten: {x.shape}")
        
        x = self.classifier(x)
        self.logger.info(f"Final output: {x.shape}")
        
        return x.squeeze(-1)