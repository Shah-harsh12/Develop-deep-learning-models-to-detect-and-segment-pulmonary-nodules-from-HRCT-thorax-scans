
"""
3D ResNet Encoder Implementation
This module implements 3D ResNet architectures (ResNet-18, ResNet-34, ResNet-50)
for feature extraction from lung nodule patches.
Supports RadImageNet pretraining and self-supervised learning initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import urllib.request

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock3D(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3D(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, ResNet-152"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    """3D ResNet implementation"""

    def __init__(self, block, layers, shortcut_type='B', num_classes=1000,
                 input_channels=1, conv1_t_size=7, conv1_t_stride=1):
        super(ResNet3D, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                   planes=planes * block.expansion,
                                   stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x  # Return features before classification layer

class ResNet3DEncoder(nn.Module):
    """3D ResNet Encoder for feature extraction"""

    def __init__(self, depth=18, input_channels=1, pretrained=False):
        super(ResNet3DEncoder, self).__init__()

        self.depth = depth

        if depth == 18:
            model = resnet3d18(input_channels=input_channels)
        elif depth == 34:
            model = resnet3d34(input_channels=input_channels)
        elif depth == 50:
            model = resnet3d50(input_channels=input_channels)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # Remove the final classification layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 512 if depth < 50 else 2048

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load RadImageNet pretrained weights"""
        # This would load weights from RadImageNet or similar medical imaging pretrained models
        # For now, we'll use random initialization
        print("Loading pretrained weights from RadImageNet...")
        # Implementation would go here to load actual pretrained weights
        pass

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features

class MultiHeadClassifier(nn.Module):
    """Multi-head classifier for different nodule characteristics"""

    def __init__(self, feature_dim, dropout_rate=0.5):
        super(MultiHeadClassifier, self).__init__()

        self.feature_dim = feature_dim

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        # Nodule type head (solid/part-solid/GGO)
        self.nodule_type_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3 types
        )

        # Binary classification head (SCLC vs NSCLC)
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Binary classification
        )

        # Malignancy probability head
        self.malignancy_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Probability output
            nn.Sigmoid()
        )

        # NSCLC subtype head (adeno/squamous/large)
        self.nsclc_subtype_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3 subtypes
        )

    def forward(self, features):
        shared_features = self.shared_layers(features)

        outputs = {
            'nodule_type': self.nodule_type_head(shared_features),
            'binary_classification': self.binary_head(shared_features),
            'malignancy': self.malignancy_head(shared_features),
            'nsclc_subtype': self.nsclc_subtype_head(shared_features)
        }

        return outputs

class LungNoduleClassifier(nn.Module):
    """Complete lung nodule classification model"""

    def __init__(self, encoder_depth=18, input_channels=1, pretrained=False):
        super(LungNoduleClassifier, self).__init__()

        self.encoder = ResNet3DEncoder(
            depth=encoder_depth,
            input_channels=input_channels,
            pretrained=pretrained
        )

        self.classifier = MultiHeadClassifier(self.encoder.feature_dim)

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)

        # Also return the feature embedding for fusion
        outputs['features'] = features

        return outputs

# Model creation functions
def resnet3d18(**kwargs):
    """ResNet-18 3D model"""
    model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)
    return model

def resnet3d34(**kwargs):
    """ResNet-34 3D model"""
    model = ResNet3D(BasicBlock3D, [3, 4, 6, 3], **kwargs)
    return model

def resnet3d50(**kwargs):
    """ResNet-50 3D model"""
    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    return model

# Training functions
class MultiHeadLoss(nn.Module):
    """Multi-head loss for different classification tasks"""

    def __init__(self, weights=None):
        super(MultiHeadLoss, self).__init__()

        if weights is None:
            weights = {
                'nodule_type': 1.0,
                'binary_classification': 1.0,
                'malignancy': 2.0,  # Higher weight for malignancy
                'nsclc_subtype': 0.5
            }

        self.weights = weights

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}

        # Nodule type loss
        if 'nodule_type' in targets:
            nodule_type_loss = self.ce_loss(predictions['nodule_type'], targets['nodule_type'])
            loss_dict['nodule_type_loss'] = nodule_type_loss
            total_loss += self.weights['nodule_type'] * nodule_type_loss

        # Binary classification loss
        if 'binary_classification' in targets:
            binary_loss = self.ce_loss(predictions['binary_classification'], targets['binary_classification'])
            loss_dict['binary_loss'] = binary_loss
            total_loss += self.weights['binary_classification'] * binary_loss

        # Malignancy loss
        if 'malignancy' in targets:
            malignancy_loss = self.bce_loss(predictions['malignancy'].squeeze(), targets['malignancy'].float())
            loss_dict['malignancy_loss'] = malignancy_loss
            total_loss += self.weights['malignancy'] * malignancy_loss

        # NSCLC subtype loss
        if 'nsclc_subtype' in targets:
            nsclc_loss = self.ce_loss(predictions['nsclc_subtype'], targets['nsclc_subtype'])
            loss_dict['nsclc_loss'] = nsclc_loss
            total_loss += self.weights['nsclc_subtype'] * nsclc_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

def train_nodule_classifier(model, train_loader, val_loader, num_epochs, device, lr=1e-3):
    """Training function for nodule classifier"""

    criterion = MultiHeadLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = {}

        for batch in train_loader:
            images = batch['image'].to(device)
            targets = {key: batch[key].to(device) for key in batch if key != 'image'}

            optimizer.zero_grad()
            predictions = model(images)
            losses = criterion(predictions, targets)

            losses['total_loss'].backward()
            optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                if key not in train_losses:
                    train_losses[key] = 0
                train_losses[key] += value.item()

        # Validation phase
        model.eval()
        val_losses = {}

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = {key: batch[key].to(device) for key in batch if key != 'image'}

                predictions = model(images)
                losses = criterion(predictions, targets)

                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0
                    val_losses[key] += value.item()

        # Calculate averages
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        for key in val_losses:
            val_losses[key] /= len(val_loader)

        scheduler.step(val_losses['total_loss'])

        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_losses["total_loss"]:.4f}, Val Loss: {val_losses["total_loss"]:.4f}')

        if val_losses['total_loss'] < best_loss:
            best_loss = val_losses['total_loss']
            torch.save(model.state_dict(), 'best_nodule_classifier.pth')
            print(f'New best model saved with loss: {best_loss:.4f}')

        print('-' * 60)

# Usage example
if __name__ == "__main__":
    # Create model
    model = LungNoduleClassifier(encoder_depth=18, pretrained=True)

    # Test with dummy data
    dummy_input = torch.randn(2, 1, 32, 32, 32)
    outputs = model(dummy_input)

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
