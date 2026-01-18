
"""
Training Scripts and Configuration for Lung Cancer Detection Pipeline
This module provides comprehensive training scripts, configuration management,
and experiment tracking for the complete pipeline.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
import wandb  # For experiment tracking
import json
from tqdm import tqdm

# Import our custom modules
from complete_pipeline import LungCancerPipeline, LungCancerDataset, create_default_config
from gradcam_utils import ModelVisualizer, LungRADSReporter

class ExperimentTracker:
    """Experiment tracking and logging"""

    def __init__(self, config, use_wandb=False):
        self.config = config
        self.use_wandb = use_wandb

        # Setup logging
        self.setup_logging()

        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project="lung-cancer-detection",
                config=config,
                name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics, step=None, prefix=""):
        """Log metrics to both local logs and wandb"""
        # Log to local logger
        metrics_str = ", ".join([f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            if step is not None:
                wandb_metrics["step"] = step
            wandb.log(wandb_metrics)

    def save_model_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['experiment']['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")

class MultiTaskTrainer:
    """Trainer for multi-task lung cancer detection"""

    def __init__(self, config, experiment_tracker):
        self.config = config
        self.tracker = experiment_tracker
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize pipeline
        self.pipeline = LungCancerPipeline(config)

        # Setup loss functions
        self.setup_losses()

    def setup_losses(self):
        """Setup loss functions for multi-task learning"""
        self.losses = {
            'malignancy': nn.BCEWithLogitsLoss(),
            'nodule_type': nn.CrossEntropyLoss(),
            'cancer_type': nn.CrossEntropyLoss()
        }

        # Loss weights
        self.loss_weights = self.config['training']['loss_weights']

    def compute_losses(self, outputs, targets):
        """Compute multi-task losses"""
        losses = {}
        total_loss = 0

        for task, criterion in self.losses.items():
            if task in targets and task in outputs:
                if task == 'malignancy':
                    loss = criterion(outputs[task].squeeze(), targets[task].float())
                else:
                    loss = criterion(outputs[task], targets[task])

                losses[f'{task}_loss'] = loss
                total_loss += self.loss_weights.get(task, 1.0) * loss

        losses['total_loss'] = total_loss
        return losses

    def compute_metrics(self, outputs, targets):
        """Compute evaluation metrics"""
        metrics = {}

        # Malignancy metrics
        if 'malignancy' in outputs and 'malignancy' in targets:
            malignancy_pred = torch.sigmoid(outputs['malignancy']).cpu().numpy()
            malignancy_true = targets['malignancy'].cpu().numpy()

            # AUC
            from sklearn.metrics import roc_auc_score, accuracy_score
            if len(np.unique(malignancy_true)) > 1:
                metrics['malignancy_auc'] = roc_auc_score(malignancy_true, malignancy_pred)

            # Accuracy
            malignancy_pred_binary = (malignancy_pred > 0.5).astype(int)
            metrics['malignancy_accuracy'] = accuracy_score(malignancy_true, malignancy_pred_binary)

        # Classification metrics
        for task in ['nodule_type', 'cancer_type']:
            if task in outputs and task in targets:
                pred = torch.softmax(outputs[task], dim=1).argmax(dim=1).cpu().numpy()
                true = targets[task].cpu().numpy()
                metrics[f'{task}_accuracy'] = accuracy_score(true, pred)

        return metrics

    def train_epoch(self, model, train_loader, optimizer, epoch):
        """Train one epoch"""
        model.train()
        epoch_losses = {}
        epoch_metrics = {}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Train')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                k: v.to(self.device) for k, v in batch.items() 
                if k in ['malignancy', 'nodule_type', 'cancer_type']
            }

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Compute losses
            losses = self.compute_losses(outputs, targets)

            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()

            # Compute metrics
            if batch_idx % self.config['training']['log_interval'] == 0:
                batch_metrics = self.compute_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item()
            })

        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])

        return epoch_losses, epoch_metrics

    def validate_epoch(self, model, val_loader, epoch):
        """Validate one epoch"""
        model.eval()
        epoch_losses = {}
        epoch_metrics = {}

        pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Val')

        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {
                    k: v.to(self.device) for k, v in batch.items() 
                    if k in ['malignancy', 'nodule_type', 'cancer_type']
                }

                # Forward pass
                outputs = model(images)

                # Compute losses
                losses = self.compute_losses(outputs, targets)

                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()

                # Compute metrics
                batch_metrics = self.compute_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)

        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)

        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])

        return epoch_losses, epoch_metrics

    def train(self, train_loader, val_loader):
        """Full training procedure"""
        model = self.pipeline.nodule_classifier
        model.to(self.device)

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['training']['scheduler_factor'],
            patience=self.config['training']['scheduler_patience']
        )

        best_val_loss = float('inf')

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train epoch
            train_losses, train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)

            # Validate epoch
            val_losses, val_metrics = self.validate_epoch(model, val_loader, epoch)

            # Update scheduler
            scheduler.step(val_losses['total_loss'])

            # Log metrics
            self.tracker.log_metrics(train_losses, epoch, prefix="train_")
            self.tracker.log_metrics(train_metrics, epoch, prefix="train_")
            self.tracker.log_metrics(val_losses, epoch, prefix="val_")
            self.tracker.log_metrics(val_metrics, epoch, prefix="val_")

            # Save checkpoint
            all_metrics = {**train_losses, **train_metrics, **val_losses, **val_metrics}
            is_best = val_losses['total_loss'] < best_val_loss

            if is_best:
                best_val_loss = val_losses['total_loss']

            self.tracker.save_model_checkpoint(
                model, optimizer, epoch, all_metrics, is_best
            )

            # Early stopping
            if hasattr(scheduler, 'num_bad_epochs'):
                if scheduler.num_bad_epochs >= self.config['training']['early_stopping_patience']:
                    self.tracker.logger.info(f"Early stopping at epoch {epoch}")
                    break

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_data_loaders(config):
    """Create training and validation data loaders"""

    # Create dataset
    dataset = LungCancerDataset(
        data_dir=config['data']['processed_dir'],
        annotations_csv=config['data']['annotations_csv'],
        clinical_csv=config['data'].get('clinical_csv'),
        patch_size=tuple(config['model']['patch_size'])
    )

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['experiment']['seed'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train Lung Cancer Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seeds for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])

    # Create output directory
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)

    # Save config
    save_config(config, os.path.join(config['experiment']['output_dir'], 'config.yaml'))

    # Initialize experiment tracker
    tracker = ExperimentTracker(config, use_wandb=args.use_wandb)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)

    tracker.logger.info(f"Training samples: {len(train_loader.dataset)}")
    tracker.logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize trainer
    trainer = MultiTaskTrainer(config, tracker)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.pipeline.nodule_classifier.load_state_dict(checkpoint['model_state_dict'])
        tracker.logger.info(f"Resumed from checkpoint: {args.resume}")

    # Start training
    tracker.logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    tracker.logger.info("Training completed!")

# Create default configuration template
def create_training_config():
    """Create comprehensive training configuration"""
    config = {
        'experiment': {
            'name': 'lung_cancer_detection',
            'output_dir': './experiments/run_001',
            'seed': 42
        },
        'data': {
            'raw_dir': '/path/to/luna16/raw',
            'processed_dir': '/path/to/luna16/processed',
            'annotations_csv': '/path/to/annotations.csv',
            'clinical_csv': '/path/to/clinical_data.csv',
            'train_split': 0.8
        },
        'model': {
            'encoder_depth': 18,
            'pretrained': True,
            'patch_size': [32, 32, 32]
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'loss_weights': {
                'malignancy': 2.0,
                'nodule_type': 1.0,
                'cancer_type': 1.0
            },
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'early_stopping_patience': 20,
            'log_interval': 10,
            'num_workers': 4
        }
    }

    return config

if __name__ == "__main__":
    # If run directly, create a default config and run training
    if len(os.sys.argv) == 1:
        # Create default config
        config = create_training_config()

        # Save default config
        os.makedirs('configs', exist_ok=True)
        save_config(config, 'configs/default_training_config.yaml')

        print("Default configuration created at: configs/default_training_config.yaml")
        print("To start training, run:")
        print("python training_scripts.py --config configs/default_training_config.yaml")
    else:
        main()
