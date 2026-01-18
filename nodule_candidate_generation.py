
"""
Nodule Candidate Generation using 3D U-Net/Detector
This module implements nodule candidate detection from lung CT scans
using both 3D U-Net and YOLO-based approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.ops import nms
import pandas as pd
import os
from scipy import ndimage
from skimage import measure

class NoduleDetector3D(nn.Module):
    """3D U-Net based nodule detector"""

    def __init__(self, in_channels=1, out_channels=2):
        super(NoduleDetector3D, self).__init__()

        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)

        # Decoder
        self.dec4 = self._make_decoder_block(1024, 512)
        self.dec3 = self._make_decoder_block(512, 256)
        self.dec2 = self._make_decoder_block(256, 128)
        self.dec1 = self._make_decoder_block(128, 64)

        # Output layer
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e4, 2))

        # Decoder path with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1

        # Final output
        output = self.final(d1)
        return output

class YOLONoduleDetector(nn.Module):
    """YOLO-based 3D nodule detector"""

    def __init__(self, num_classes=1, num_anchors=5):
        super(YOLONoduleDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone
        self.backbone = self._make_backbone()

        # Detection head
        self.detection_head = nn.Conv3d(
            512, 
            num_anchors * (5 + num_classes),  # 5 = x, y, z, w, h, d, confidence
            kernel_size=1
        )

    def _make_backbone(self):
        return nn.Sequential(
            # Layer 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            # Layer 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            # Layer 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            # Layer 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            # Layer 5
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections

class NoduleCandidateDataset(Dataset):
    """Dataset for nodule candidate generation"""

    def __init__(self, data_dir, annotations_csv, patch_size=(64, 64, 64), transform=None):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.transform = transform

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)
        self.cases = self.annotations['seriesuid'].unique()

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id = self.cases[idx]

        # Load image
        image_path = os.path.join(self.data_dir, f'{case_id}_clean.npy')
        image = np.load(image_path).astype(np.float32)

        # Get annotations for this case
        case_annotations = self.annotations[self.annotations['seriesuid'] == case_id]

        # Create nodule mask
        nodule_mask = np.zeros_like(image)
        bboxes = []

        for _, row in case_annotations.iterrows():
            coord_x, coord_y, coord_z = int(row['coordX']), int(row['coordY']), int(row['coordZ'])
            diameter = row['diameter_mm']

            # Create spherical mask for nodule
            radius = int(diameter / 2)
            z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
            mask = ((x - coord_x)**2 + (y - coord_y)**2 + (z - coord_z)**2) <= radius**2
            nodule_mask[mask] = 1

            # Store bounding box
            bboxes.append([
                max(0, coord_x - radius), max(0, coord_y - radius), max(0, coord_z - radius),
                min(image.shape[2], coord_x + radius), 
                min(image.shape[1], coord_y + radius), 
                min(image.shape[0], coord_z + radius)
            ])

        # Add channel dimension
        image = image[np.newaxis, ...]
        nodule_mask = nodule_mask[np.newaxis, ...]

        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(nodule_mask),
            'bboxes': bboxes,
            'case_id': case_id
        }

class NoduleCandidateGenerator:
    """Main class for generating nodule candidates"""

    def __init__(self, model_path=None, device='cuda', threshold=0.5):
        self.device = device
        self.threshold = threshold

        # Load model
        self.model = NoduleDetector3D()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def detect_candidates(self, image):
        """
        Detect nodule candidates from 3D image

        Args:
            image (np.ndarray): 3D CT image

        Returns:
            list: List of candidate coordinates and scores
        """
        # Convert to tensor
        if len(image.shape) == 3:
            image = image[np.newaxis, np.newaxis, ...]  # Add batch and channel dims

        image_tensor = torch.from_numpy(image).float().to(self.device)

        with torch.no_grad():
            prediction = self.model(image_tensor)
            prob_map = torch.softmax(prediction, dim=1)[:, 1]  # Get nodule probability

        # Convert back to numpy
        prob_map = prob_map.cpu().numpy().squeeze()

        # Find candidates using connected components
        binary_map = (prob_map > self.threshold).astype(np.uint8)
        labeled_map = measure.label(binary_map)

        candidates = []
        for region in measure.regionprops(labeled_map):
            if region.area > 8:  # Filter small regions
                centroid = region.centroid
                bbox = region.bbox
                confidence = prob_map[labeled_map == region.label].mean()

                candidates.append({
                    'coord_z': int(centroid[0]),
                    'coord_y': int(centroid[1]),
                    'coord_x': int(centroid[2]),
                    'bbox': bbox,
                    'confidence': confidence,
                    'area': region.area
                })

        # Sort by confidence
        candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)

        return candidates

    def non_maximum_suppression_3d(self, candidates, iou_threshold=0.3):
        """Apply 3D Non-Maximum Suppression"""

        if len(candidates) == 0:
            return candidates

        # Convert to format suitable for NMS
        boxes = []
        scores = []

        for candidate in candidates:
            bbox = candidate['bbox']
            # Convert (min_z, min_y, min_x, max_z, max_y, max_x) format
            box = [bbox[2], bbox[1], bbox[0], bbox[5], bbox[4], bbox[3]]  # x1,y1,z1,x2,y2,z2
            boxes.append(box[:4])  # Use only x,y coordinates for 2D NMS approximation
            scores.append(candidate['confidence'])

        boxes = torch.tensor(boxes).float()
        scores = torch.tensor(scores).float()

        # Apply NMS
        keep_indices = nms(boxes, scores, iou_threshold)

        # Return filtered candidates
        filtered_candidates = [candidates[i] for i in keep_indices.tolist()]

        return filtered_candidates

    def extract_roi_patches(self, image, candidates, patch_size=(32, 32, 32)):
        """Extract ROI patches around candidates"""

        patches = []
        valid_candidates = []

        for candidate in candidates:
            coord_x = candidate['coord_x']
            coord_y = candidate['coord_y'] 
            coord_z = candidate['coord_z']

            # Calculate patch boundaries
            half_x, half_y, half_z = patch_size[2]//2, patch_size[1]//2, patch_size[0]//2

            start_x = max(0, coord_x - half_x)
            end_x = min(image.shape[2], coord_x + half_x)
            start_y = max(0, coord_y - half_y)
            end_y = min(image.shape[1], coord_y + half_y)
            start_z = max(0, coord_z - half_z)
            end_z = min(image.shape[0], coord_z + half_z)

            # Extract patch
            patch = image[start_z:end_z, start_y:end_y, start_x:end_x]

            # Ensure patch has correct size by padding if necessary
            if patch.shape != patch_size:
                padded_patch = np.zeros(patch_size, dtype=patch.dtype)

                # Calculate padding offsets
                z_offset = (patch_size[0] - patch.shape[0]) // 2
                y_offset = (patch_size[1] - patch.shape[1]) // 2
                x_offset = (patch_size[2] - patch.shape[2]) // 2

                # Place patch in center of padded array
                padded_patch[
                    z_offset:z_offset + patch.shape[0],
                    y_offset:y_offset + patch.shape[1],
                    x_offset:x_offset + patch.shape[2]
                ] = patch

                patch = padded_patch

            patches.append(patch)
            valid_candidates.append(candidate)

        return patches, valid_candidates

def train_nodule_detector(model, train_loader, val_loader, num_epochs, device, lr=1e-3):
    """Training function for nodule detection"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).long()

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_nodule_detector.pth')
            print(f'New best model saved with loss: {best_loss:.4f}')

        print('-' * 60)

# Usage example
if __name__ == "__main__":
    # Initialize candidate generator
    generator = NoduleCandidateGenerator(
        model_path='best_nodule_detector.pth',
        device='cuda',
        threshold=0.5
    )

    # Load and process image
    image = np.load('/path/to/processed_image.npy')
    candidates = generator.detect_candidates(image)

    # Apply NMS
    filtered_candidates = generator.non_maximum_suppression_3d(candidates)

    # Extract ROI patches
    patches, valid_candidates = generator.extract_roi_patches(
        image, filtered_candidates, patch_size=(32, 32, 32)
    )

    print(f"Found {len(valid_candidates)} nodule candidates")
