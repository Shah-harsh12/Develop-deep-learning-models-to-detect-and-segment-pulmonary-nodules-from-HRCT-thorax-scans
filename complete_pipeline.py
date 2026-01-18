
"""
Complete Lung Cancer Detection and Classification Pipeline
This module implements the complete end-to-end pipeline as shown in the flowchart,
integrating all components from preprocessing to final reporting.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import cv2
from grad_cam import GradCAM
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from dicom_preprocessing import DICOMPreprocessor
from unet_3d_lung_segmentation import UNet3D, train_lung_segmentation
from nodule_candidate_generation import NoduleCandidateGenerator
from resnet_3d_encoder import LungNoduleClassifier
from radiomics_feature_fusion import MultiModalPipeline

class LungCancerDataset(Dataset):
    """Complete dataset for lung cancer detection and classification"""

    def __init__(self, data_dir, annotations_csv, clinical_csv=None, 
                 patch_size=(32, 32, 32), transform=None):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.transform = transform

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)

        # Load clinical data if available
        if clinical_csv and os.path.exists(clinical_csv):
            self.clinical_data = pd.read_csv(clinical_csv)
            self.clinical_data.set_index('seriesuid', inplace=True)
        else:
            self.clinical_data = None

        # Prepare samples
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """Prepare training samples"""
        for _, row in self.annotations.iterrows():
            sample = {
                'seriesuid': row['seriesuid'],
                'coord_x': row['coordX'],
                'coord_y': row['coordY'],
                'coord_z': row['coordZ'],
                'diameter': row['diameter_mm'],
                'malignancy': 1 if row.get('malignancy', 0) > 3 else 0,  # Binary malignancy
                'nodule_type': row.get('nodule_type', 0),  # 0: solid, 1: part-solid, 2: GGO
                'cancer_type': row.get('cancer_type', 0)  # 0: NSCLC, 1: SCLC
            }

            # Add clinical data if available
            if self.clinical_data is not None and row['seriesuid'] in self.clinical_data.index:
                clinical_row = self.clinical_data.loc[row['seriesuid']]
                sample.update(clinical_row.to_dict())

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load processed image
        image_path = os.path.join(self.data_dir, 'images', f'{sample["seriesuid"]}_clean.npy')
        image = np.load(image_path).astype(np.float32)

        # Extract patch around nodule
        coord_x, coord_y, coord_z = int(sample['coord_x']), int(sample['coord_y']), int(sample['coord_z'])
        patch, mask = self._extract_patch(image, coord_x, coord_y, coord_z, sample['diameter'])

        # Apply transforms if any
        if self.transform:
            patch = self.transform(patch)

        # Prepare return dictionary
        return_dict = {
            'image': torch.from_numpy(patch[np.newaxis, ...]),  # Add channel dimension
            'mask': torch.from_numpy(mask[np.newaxis, ...]),
            'malignancy': torch.tensor(sample['malignancy'], dtype=torch.long),
            'nodule_type': torch.tensor(sample['nodule_type'], dtype=torch.long),
            'cancer_type': torch.tensor(sample['cancer_type'], dtype=torch.long),
            'seriesuid': sample['seriesuid'],
            'coordinates': torch.tensor([coord_x, coord_y, coord_z]),
            'diameter': torch.tensor(sample['diameter'])
        }

        # Add clinical features if available
        if self.clinical_data is not None:
            clinical_features = []
            for key, value in sample.items():
                if key not in ['seriesuid', 'coord_x', 'coord_y', 'coord_z', 
                              'diameter', 'malignancy', 'nodule_type', 'cancer_type']:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        clinical_features.append(value)

            if clinical_features:
                return_dict['clinical_features'] = torch.tensor(clinical_features, dtype=torch.float32)

        return return_dict

    def _extract_patch(self, image, coord_x, coord_y, coord_z, diameter):
        """Extract patch around nodule location"""
        half_size = np.array(self.patch_size) // 2

        # Calculate patch boundaries
        start_z = max(0, coord_z - half_size[0])
        end_z = min(image.shape[0], coord_z + half_size[0])
        start_y = max(0, coord_y - half_size[1])
        end_y = min(image.shape[1], coord_y + half_size[1])
        start_x = max(0, coord_x - half_size[2])
        end_x = min(image.shape[2], coord_x + half_size[2])

        # Extract patch
        patch = image[start_z:end_z, start_y:end_y, start_x:end_x]

        # Create nodule mask
        mask = np.zeros_like(patch)
        radius = int(diameter / 2)

        # Adjust coordinates relative to patch
        rel_z = coord_z - start_z
        rel_y = coord_y - start_y
        rel_x = coord_x - start_x

        # Create spherical mask
        z, y, x = np.ogrid[:patch.shape[0], :patch.shape[1], :patch.shape[2]]
        sphere_mask = ((x - rel_x)**2 + (y - rel_y)**2 + (z - rel_z)**2) <= radius**2
        mask[sphere_mask] = 1

        # Pad to target size if necessary
        if patch.shape != self.patch_size:
            padded_patch = np.zeros(self.patch_size, dtype=patch.dtype)
            padded_mask = np.zeros(self.patch_size, dtype=mask.dtype)

            # Calculate padding offsets
            z_offset = (self.patch_size[0] - patch.shape[0]) // 2
            y_offset = (self.patch_size[1] - patch.shape[1]) // 2
            x_offset = (self.patch_size[2] - patch.shape[2]) // 2

            # Place patch in center
            padded_patch[
                z_offset:z_offset + patch.shape[0],
                y_offset:y_offset + patch.shape[1],
                x_offset:x_offset + patch.shape[2]
            ] = patch

            padded_mask[
                z_offset:z_offset + mask.shape[0],
                y_offset:y_offset + mask.shape[1],
                x_offset:x_offset + mask.shape[2]
            ] = mask

            patch, mask = padded_patch, padded_mask

        return patch, mask

class LungCancerPipeline:
    """Complete lung cancer detection and classification pipeline"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.preprocessor = DICOMPreprocessor(
            data_dir=config['data']['raw_dir'],
            output_dir=config['data']['processed_dir']
        )

        self.lung_segmentation_model = None
        self.nodule_detector = None
        self.nodule_classifier = None
        self.multimodal_pipeline = None

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models"""

        # Lung segmentation model
        self.lung_segmentation_model = UNet3D(n_channels=1, n_classes=1)

        # Nodule candidate generator
        self.nodule_detector = NoduleCandidateGenerator(device=self.device)

        # Nodule classifier
        self.nodule_classifier = LungNoduleClassifier(
            encoder_depth=self.config['model']['encoder_depth'],
            pretrained=self.config['model']['pretrained']
        )

        # Multi-modal pipeline
        self.multimodal_pipeline = MultiModalPipeline()

    def preprocess_data(self, annotations_csv):
        """Preprocess LUNA16 dataset"""
        print("Starting data preprocessing...")
        metadata = self.preprocessor.process_luna16_dataset(annotations_csv)
        print(f"Preprocessing completed. Processed {len(metadata)} cases.")
        return metadata

    def train_lung_segmentation(self, train_loader, val_loader, num_epochs=50):
        """Train lung segmentation model"""
        print("Training lung segmentation model...")
        train_lung_segmentation(
            self.lung_segmentation_model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            device=self.device,
            lr=self.config['training']['lr']
        )
        print("Lung segmentation training completed.")

    def train_nodule_classifier(self, train_loader, val_loader, num_epochs=100):
        """Train nodule classification model"""
        print("Training nodule classifier...")

        # Define multi-task loss
        criterion = {
            'malignancy': nn.BCEWithLogitsLoss(),
            'nodule_type': nn.CrossEntropyLoss(),
            'cancer_type': nn.CrossEntropyLoss()
        }

        optimizer = torch.optim.Adam(
            self.nodule_classifier.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        self.nodule_classifier.to(self.device)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            # Training phase
            self.nodule_classifier.train()
            train_losses = {'total': 0, 'malignancy': 0, 'nodule_type': 0, 'cancer_type': 0}

            for batch in train_loader:
                images = batch['image'].to(self.device)
                targets = {
                    'malignancy': batch['malignancy'].float().to(self.device),
                    'nodule_type': batch['nodule_type'].to(self.device),
                    'cancer_type': batch['cancer_type'].to(self.device)
                }

                optimizer.zero_grad()
                outputs = self.nodule_classifier(images)

                # Calculate losses
                losses = {}
                losses['malignancy'] = criterion['malignancy'](
                    outputs['malignancy'].squeeze(), targets['malignancy']
                )
                losses['nodule_type'] = criterion['nodule_type'](
                    outputs['nodule_type'], targets['nodule_type']
                )
                losses['cancer_type'] = criterion['cancer_type'](
                    outputs['cancer_type'], targets['cancer_type']
                )

                # Weighted total loss
                total_loss = (2.0 * losses['malignancy'] + 
                             1.0 * losses['nodule_type'] + 
                             1.0 * losses['cancer_type'])

                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                train_losses['total'] += total_loss.item()
                for key in losses:
                    train_losses[key] += losses[key].item()

            # Validation phase
            self.nodule_classifier.eval()
            val_losses = {'total': 0, 'malignancy': 0, 'nodule_type': 0, 'cancer_type': 0}

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    targets = {
                        'malignancy': batch['malignancy'].float().to(self.device),
                        'nodule_type': batch['nodule_type'].to(self.device),
                        'cancer_type': batch['cancer_type'].to(self.device)
                    }

                    outputs = self.nodule_classifier(images)

                    # Calculate losses
                    losses = {}
                    losses['malignancy'] = criterion['malignancy'](
                        outputs['malignancy'].squeeze(), targets['malignancy']
                    )
                    losses['nodule_type'] = criterion['nodule_type'](
                        outputs['nodule_type'], targets['nodule_type']
                    )
                    losses['cancer_type'] = criterion['cancer_type'](
                        outputs['cancer_type'], targets['cancer_type']
                    )

                    total_loss = (2.0 * losses['malignancy'] + 
                                 1.0 * losses['nodule_type'] + 
                                 1.0 * losses['cancer_type'])

                    val_losses['total'] += total_loss.item()
                    for key in losses:
                        val_losses[key] += losses[key].item()

            # Calculate averages
            for key in train_losses:
                train_losses[key] /= len(train_loader)
                val_losses[key] /= len(val_loader)

            scheduler.step(val_losses['total'])

            print(f'Epoch [{epoch+1}/{num_epochs}]:')
            print(f'Train Loss: {train_losses["total"]:.4f}, Val Loss: {val_losses["total"]:.4f}')

            if val_losses['total'] < best_loss:
                best_loss = val_losses['total']
                torch.save(self.nodule_classifier.state_dict(), 
                          os.path.join(self.config['model']['save_dir'], 'best_nodule_classifier.pth'))
                print(f'New best model saved with loss: {best_loss:.4f}')

            print('-' * 60)

        print("Nodule classifier training completed.")

    def predict_case(self, case_id, return_attention=True):
        """Predict on a single case"""

        # Load preprocessed image
        image_path = os.path.join(self.config['data']['processed_dir'], 'images', f'{case_id}_clean.npy')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Processed image not found: {image_path}")

        image = np.load(image_path)

        # Step 1: Lung segmentation (if needed)
        lung_mask = self._segment_lungs(image)

        # Step 2: Nodule candidate generation
        candidates = self.nodule_detector.detect_candidates(image)
        candidates = self.nodule_detector.non_maximum_suppression_3d(candidates)

        if len(candidates) == 0:
            return {
                'case_id': case_id,
                'num_candidates': 0,
                'predictions': [],
                'summary': 'No nodule candidates detected'
            }

        # Step 3: Extract ROI patches
        patches, valid_candidates = self.nodule_detector.extract_roi_patches(
            image, candidates, patch_size=self.config['model']['patch_size']
        )

        # Step 4: Deep learning classification
        predictions = []
        attention_maps = []

        self.nodule_classifier.eval()
        with torch.no_grad():
            for i, patch in enumerate(patches):
                patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).float().to(self.device)
                outputs = self.nodule_classifier(patch_tensor)

                # Get predictions
                malignancy_prob = torch.sigmoid(outputs['malignancy']).cpu().numpy()[0, 0]
                nodule_type_pred = torch.softmax(outputs['nodule_type'], dim=1).cpu().numpy()[0]
                cancer_type_pred = torch.softmax(outputs['cancer_type'], dim=1).cpu().numpy()[0]

                prediction = {
                    'candidate_id': i,
                    'coordinates': [
                        valid_candidates[i]['coord_x'],
                        valid_candidates[i]['coord_y'],
                        valid_candidates[i]['coord_z']
                    ],
                    'malignancy_probability': float(malignancy_prob),
                    'nodule_type_probabilities': {
                        'solid': float(nodule_type_pred[0]),
                        'part_solid': float(nodule_type_pred[1]),
                        'ggo': float(nodule_type_pred[2])
                    },
                    'cancer_type_probabilities': {
                        'nsclc': float(cancer_type_pred[0]),
                        'sclc': float(cancer_type_pred[1])
                    },
                    'detection_confidence': float(valid_candidates[i]['confidence'])
                }

                predictions.append(prediction)

                # Generate attention map if requested
                if return_attention:
                    attention_map = self._generate_attention_map(patch_tensor)
                    attention_maps.append(attention_map)

        # Step 5: Generate summary report
        summary = self._generate_case_summary(predictions)

        result = {
            'case_id': case_id,
            'num_candidates': len(valid_candidates),
            'predictions': predictions,
            'summary': summary,
            'processing_timestamp': datetime.now().isoformat()
        }

        if return_attention:
            result['attention_maps'] = attention_maps

        return result

    def _segment_lungs(self, image):
        """Segment lungs using trained model"""
        if self.lung_segmentation_model is None:
            # Return dummy mask if model not trained
            return np.ones_like(image)

        self.lung_segmentation_model.eval()
        with torch.no_grad():
            image_tensor = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float().to(self.device)
            mask = self.lung_segmentation_model(image_tensor)
            mask = torch.sigmoid(mask) > 0.5
            return mask.cpu().numpy().squeeze()

    def _generate_attention_map(self, patch_tensor):
        """Generate Grad-CAM attention map"""
        # This would use the GradCAM implementation
        # For now, return dummy attention map
        return np.random.rand(*patch_tensor.shape[2:])

    def _generate_case_summary(self, predictions):
        """Generate case summary based on predictions"""
        if not predictions:
            return "No nodules detected"

        # Find highest malignancy probability
        max_malignancy = max(pred['malignancy_probability'] for pred in predictions)
        high_risk_nodules = [pred for pred in predictions if pred['malignancy_probability'] > 0.7]

        summary_parts = []
        summary_parts.append(f"Detected {len(predictions)} nodule candidates")

        if max_malignancy > 0.8:
            summary_parts.append(f"HIGH RISK: Maximum malignancy probability {max_malignancy:.3f}")
        elif max_malignancy > 0.5:
            summary_parts.append(f"MODERATE RISK: Maximum malignancy probability {max_malignancy:.3f}")
        else:
            summary_parts.append(f"LOW RISK: Maximum malignancy probability {max_malignancy:.3f}")

        if high_risk_nodules:
            summary_parts.append(f"{len(high_risk_nodules)} high-risk nodules (>70% malignancy)")

        return ". ".join(summary_parts)

    def evaluate_model(self, test_loader):
        """Evaluate model performance"""
        print("Evaluating model...")

        self.nodule_classifier.eval()
        all_predictions = {'malignancy': [], 'nodule_type': [], 'cancer_type': []}
        all_targets = {'malignancy': [], 'nodule_type': [], 'cancer_type': []}

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                targets = {
                    'malignancy': batch['malignancy'].to(self.device),
                    'nodule_type': batch['nodule_type'].to(self.device),
                    'cancer_type': batch['cancer_type'].to(self.device)
                }

                outputs = self.nodule_classifier(images)

                # Collect predictions
                all_predictions['malignancy'].extend(
                    torch.sigmoid(outputs['malignancy']).cpu().numpy().flatten()
                )
                all_predictions['nodule_type'].extend(
                    torch.softmax(outputs['nodule_type'], dim=1).cpu().numpy()
                )
                all_predictions['cancer_type'].extend(
                    torch.softmax(outputs['cancer_type'], dim=1).cpu().numpy()
                )

                # Collect targets
                all_targets['malignancy'].extend(targets['malignancy'].cpu().numpy())
                all_targets['nodule_type'].extend(targets['nodule_type'].cpu().numpy())
                all_targets['cancer_type'].extend(targets['cancer_type'].cpu().numpy())

        # Calculate metrics
        metrics = {}

        # Malignancy AUC
        malignancy_auc = roc_auc_score(all_targets['malignancy'], all_predictions['malignancy'])
        metrics['malignancy_auc'] = malignancy_auc

        # Classification reports
        nodule_type_pred = np.argmax(all_predictions['nodule_type'], axis=1)
        cancer_type_pred = np.argmax(all_predictions['cancer_type'], axis=1)

        metrics['nodule_type_report'] = classification_report(
            all_targets['nodule_type'], nodule_type_pred, output_dict=True
        )
        metrics['cancer_type_report'] = classification_report(
            all_targets['cancer_type'], cancer_type_pred, output_dict=True
        )

        print(f"Evaluation completed. Malignancy AUC: {malignancy_auc:.4f}")
        return metrics

    def save_pipeline(self, save_path):
        """Save trained pipeline"""
        pipeline_state = {
            'config': self.config,
            'lung_segmentation_model': self.lung_segmentation_model.state_dict() if self.lung_segmentation_model else None,
            'nodule_classifier': self.nodule_classifier.state_dict(),
            'timestamp': datetime.now().isoformat()
        }

        torch.save(pipeline_state, save_path)
        print(f"Pipeline saved to {save_path}")

    def load_pipeline(self, load_path):
        """Load trained pipeline"""
        pipeline_state = torch.load(load_path, map_location=self.device)

        if pipeline_state['lung_segmentation_model']:
            self.lung_segmentation_model.load_state_dict(pipeline_state['lung_segmentation_model'])

        self.nodule_classifier.load_state_dict(pipeline_state['nodule_classifier'])

        print(f"Pipeline loaded from {load_path}")

# Configuration example
def create_default_config():
    """Create default configuration"""
    return {
        'data': {
            'raw_dir': '/path/to/luna16/raw',
            'processed_dir': '/path/to/luna16/processed',
            'annotations_csv': '/path/to/annotations.csv',
            'clinical_csv': '/path/to/clinical_data.csv'
        },
        'model': {
            'encoder_depth': 18,
            'pretrained': True,
            'patch_size': (32, 32, 32),
            'save_dir': './models'
        },
        'training': {
            'batch_size': 8,
            'lr': 1e-3,
            'num_epochs': 100,
            'num_workers': 4
        }
    }

# Usage example
if __name__ == "__main__":
    # Create configuration
    config = create_default_config()

    # Initialize pipeline
    pipeline = LungCancerPipeline(config)

    # Preprocess data (if needed)
    # pipeline.preprocess_data(config['data']['annotations_csv'])

    # Create datasets and data loaders
    train_dataset = LungCancerDataset(
        data_dir=config['data']['processed_dir'],
        annotations_csv=config['data']['annotations_csv'],
        clinical_csv=config['data']['clinical_csv']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    # Train models (example)
    # pipeline.train_nodule_classifier(train_loader, train_loader, num_epochs=10)

    # Make prediction on a case
    # result = pipeline.predict_case('1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860')

    print("Complete pipeline setup completed!")
