
"""
GradCAM Implementation and Visualization Utilities
This module provides GradCAM visualization for 3D CNNs and other utility functions
for model interpretability and reporting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
from datetime import datetime
import os
import json

class GradCAM3D:
    """GradCAM for 3D CNNs"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate GradCAM heatmap

        Args:
            input_tensor (torch.Tensor): Input tensor (1, C, D, H, W)
            class_idx (int): Target class index for gradient computation

        Returns:
            np.ndarray: GradCAM heatmap
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Handle multi-head output
        if isinstance(output, dict):
            if 'malignancy' in output:
                target_output = output['malignancy']
            else:
                target_output = list(output.values())[0]
        else:
            target_output = output

        # Get the target class score
        if class_idx is None:
            if target_output.dim() == 1:
                target_score = target_output[0]
            else:
                target_score = torch.max(target_output)
        else:
            target_score = target_output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]  # (C, D, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # (C, D, H, W)

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2, 3))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (D, H, W)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

class LungRADSReporter:
    """Generate Lung-RADS compatible reports"""

    def __init__(self):
        self.lung_rads_categories = {
            1: {"description": "No nodules or definitely benign", "management": "Continue annual screening"},
            2: {"description": "Benign appearance", "management": "Continue annual screening"},
            3: {"description": "Probably benign", "management": "6-month follow-up CT"},
            4A: {"description": "Suspicious", "management": "3-month follow-up CT or PET/CT"},
            4B: {"description": "Very suspicious", "management": "1-month follow-up CT or PET/CT, consider biopsy"},
            4X: {"description": "Very suspicious with additional features", "management": "Consider chest CT with contrast, PET/CT, or tissue sampling"}
        }

    def categorize_nodule(self, prediction):
        """
        Categorize nodule according to Lung-RADS

        Args:
            prediction (dict): Nodule prediction results

        Returns:
            dict: Lung-RADS category and management
        """
        malignancy_prob = prediction['malignancy_probability']
        detection_conf = prediction['detection_confidence']

        # Simple categorization based on malignancy probability
        # In practice, this would consider size, growth, morphology, etc.

        if malignancy_prob < 0.1:
            category = 1 if detection_conf > 0.8 else 2
        elif malignancy_prob < 0.3:
            category = 2
        elif malignancy_prob < 0.5:
            category = 3
        elif malignancy_prob < 0.7:
            category = "4A"
        elif malignancy_prob < 0.9:
            category = "4B"
        else:
            category = "4X"

        return {
            'category': category,
            'description': self.lung_rads_categories[category]['description'],
            'management': self.lung_rads_categories[category]['management'],
            'probability': malignancy_prob
        }

    def generate_report(self, case_result):
        """Generate comprehensive report"""

        report_sections = []

        # Header
        report_sections.append("="*60)
        report_sections.append("LUNG CANCER DETECTION REPORT")
        report_sections.append("="*60)
        report_sections.append(f"Case ID: {case_result['case_id']}")
        report_sections.append(f"Analysis Date: {case_result.get('processing_timestamp', datetime.now().isoformat())}")
        report_sections.append("")

        # Summary
        report_sections.append("EXECUTIVE SUMMARY:")
        report_sections.append("-" * 20)
        report_sections.append(case_result['summary'])
        report_sections.append("")

        # Detailed findings
        report_sections.append("DETAILED FINDINGS:")
        report_sections.append("-" * 20)

        if not case_result['predictions']:
            report_sections.append("No nodule candidates detected.")
        else:
            for i, pred in enumerate(case_result['predictions']):
                lung_rads = self.categorize_nodule(pred)

                report_sections.append(f"Nodule #{i+1}:")
                report_sections.append(f"  Location: ({pred['coordinates'][0]}, {pred['coordinates'][1]}, {pred['coordinates'][2]})")
                report_sections.append(f"  Malignancy Probability: {pred['malignancy_probability']:.3f}")
                report_sections.append(f"  Detection Confidence: {pred['detection_confidence']:.3f}")
                report_sections.append(f"  Nodule Type: {max(pred['nodule_type_probabilities'], key=pred['nodule_type_probabilities'].get)}")
                report_sections.append(f"  Cancer Type: {max(pred['cancer_type_probabilities'], key=pred['cancer_type_probabilities'].get)}")
                report_sections.append(f"  Lung-RADS Category: {lung_rads['category']}")
                report_sections.append(f"  Management: {lung_rads['management']}")
                report_sections.append("")

        # Recommendations
        report_sections.append("RECOMMENDATIONS:")
        report_sections.append("-" * 20)

        high_risk_count = len([p for p in case_result['predictions'] if p['malignancy_probability'] > 0.7])

        if high_risk_count > 0:
            report_sections.append("URGENT: High-risk nodules detected. Recommend immediate clinical correlation.")
        elif len(case_result['predictions']) > 0:
            report_sections.append("Follow-up imaging recommended based on Lung-RADS guidelines above.")
        else:
            report_sections.append("Continue routine screening as per guidelines.")

        report_sections.append("")
        report_sections.append("="*60)
        report_sections.append("END OF REPORT")
        report_sections.append("="*60)

        return "\n".join(report_sections)

class ModelVisualizer:
    """Visualization utilities for model outputs"""

    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_attention_overlay(self, image_slice, attention_slice, alpha=0.4):
        """Plot image with attention overlay"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention map
        axes[1].imshow(attention_slice, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image_slice, cmap='gray')
        axes[2].imshow(attention_slice, cmap='jet', alpha=alpha)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

    def plot_nodule_detection_results(self, image_slice, detections, slice_idx):
        """Plot detection results on image slice"""

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image_slice, cmap='gray')

        # Plot detections
        for i, detection in enumerate(detections):
            coord_x, coord_y, coord_z = detection['coordinates']

            # Check if detection is in current slice (within ±2 slices)
            if abs(coord_z - slice_idx) <= 2:
                # Draw circle around detection
                circle = patches.Circle(
                    (coord_x, coord_y), 
                    radius=10,
                    linewidth=2,
                    edgecolor='red' if detection['malignancy_probability'] > 0.5 else 'yellow',
                    facecolor='none'
                )
                ax.add_patch(circle)

                # Add label
                ax.text(coord_x + 15, coord_y, 
                       f"{detection['malignancy_probability']:.2f}",
                       color='white', fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='red' if detection['malignancy_probability'] > 0.5 else 'yellow', alpha=0.7))

        ax.set_title(f'Nodule Detections - Slice {slice_idx}')
        ax.axis('off')

        return fig

    def plot_roc_curves(self, y_true, y_scores, labels):
        """Plot ROC curves for multi-class classification"""

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (true_labels, scores, label) in enumerate(zip(y_true, y_scores, labels)):
            fpr, tpr, _ = roc_curve(true_labels, scores)
            auc_score = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=self.colors[i], 
                   label=f'{label} (AUC = {auc_score:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_precision_recall_curves(self, y_true, y_scores, labels):
        """Plot precision-recall curves"""

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (true_labels, scores, label) in enumerate(zip(y_true, y_scores, labels)):
            precision, recall, _ = precision_recall_curve(true_labels, scores)
            auc_score = auc(recall, precision)

            ax.plot(recall, precision, color=self.colors[i],
                   label=f'{label} (AUC = {auc_score:.3f})', linewidth=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_feature_importance(self, feature_names, importance_scores, top_n=20):
        """Plot feature importance"""

        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, importance_scores[sorted_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')

        plt.tight_layout()
        return fig

    def create_dashboard(self, case_result, save_path=None):
        """Create comprehensive visualization dashboard"""

        fig = plt.figure(figsize=(20, 12))

        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f"Lung Cancer Detection Dashboard - Case: {case_result['case_id']}", 
                    fontsize=16, fontweight='bold')

        # Summary statistics
        ax1 = fig.add_subplot(gs[0, 0])
        stats_text = f"""
        Nodule Candidates: {case_result['num_candidates']}

        High Risk (>70%): {len([p for p in case_result['predictions'] if p['malignancy_probability'] > 0.7])}

        Moderate Risk (30-70%): {len([p for p in case_result['predictions'] if 0.3 < p['malignancy_probability'] <= 0.7])}

        Low Risk (<30%): {len([p for p in case_result['predictions'] if p['malignancy_probability'] <= 0.3])}
        """
        ax1.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax1.set_title('Detection Summary')
        ax1.axis('off')

        # Malignancy probability distribution
        if case_result['predictions']:
            ax2 = fig.add_subplot(gs[0, 1])
            probs = [p['malignancy_probability'] for p in case_result['predictions']]
            ax2.hist(probs, bins=10, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0.5, color='red', linestyle='--', label='Malignancy Threshold')
            ax2.set_xlabel('Malignancy Probability')
            ax2.set_ylabel('Number of Nodules')
            ax2.set_title('Malignancy Distribution')
            ax2.legend()

        # Nodule type distribution
        if case_result['predictions']:
            ax3 = fig.add_subplot(gs[0, 2])
            nodule_types = []
            for pred in case_result['predictions']:
                nodule_types.append(max(pred['nodule_type_probabilities'], 
                                      key=pred['nodule_type_probabilities'].get))

            type_counts = pd.Series(nodule_types).value_counts()
            ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            ax3.set_title('Nodule Type Distribution')

        # Risk stratification
        ax4 = fig.add_subplot(gs[0, 3])
        if case_result['predictions']:
            risk_levels = []
            for pred in case_result['predictions']:
                prob = pred['malignancy_probability']
                if prob > 0.7:
                    risk_levels.append('High')
                elif prob > 0.3:
                    risk_levels.append('Moderate')
                else:
                    risk_levels.append('Low')

            risk_counts = pd.Series(risk_levels).value_counts()
            colors = ['red', 'orange', 'green']
            ax4.bar(risk_counts.index, risk_counts.values, color=colors[:len(risk_counts)])
            ax4.set_ylabel('Number of Nodules')
            ax4.set_title('Risk Stratification')
            ax4.tick_params(axis='x', rotation=45)

        # Text report in remaining space
        ax5 = fig.add_subplot(gs[1:, :])
        report_text = case_result['summary']
        ax5.text(0.05, 0.95, report_text, fontsize=10, verticalalignment='top',
                transform=ax5.transAxes, wrap=True)
        ax5.set_title('Detailed Report')
        ax5.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

# Utility functions
def save_results_to_json(results, output_path):
    """Save prediction results to JSON file"""

    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    converted_results = convert_numpy_types(results)

    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)

    print(f"Results saved to {output_path}")

def create_batch_report(results_list, output_dir):
    """Create batch processing report"""

    os.makedirs(output_dir, exist_ok=True)

    # Aggregate statistics
    total_cases = len(results_list)
    total_nodules = sum(len(r['predictions']) for r in results_list)
    high_risk_cases = len([r for r in results_list 
                          if any(p['malignancy_probability'] > 0.7 for p in r['predictions'])])

    # Create summary report
    summary_report = f"""
    BATCH PROCESSING SUMMARY
    ========================

    Total Cases Processed: {total_cases}
    Total Nodules Detected: {total_nodules}
    High-Risk Cases: {high_risk_cases} ({high_risk_cases/total_cases*100:.1f}%)

    Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    with open(os.path.join(output_dir, 'batch_summary.txt'), 'w') as f:
        f.write(summary_report)

    # Save detailed results
    save_results_to_json(results_list, os.path.join(output_dir, 'detailed_results.json'))

    print(f"Batch report saved to {output_dir}")

# Usage example
if __name__ == "__main__":
    # Example usage of GradCAM
    print("GradCAM and visualization utilities ready!")

    # This would typically be used within the main pipeline
    # gradcam = GradCAM3D(model, 'encoder.layer4.1.conv2')
    # cam = gradcam.generate_cam(input_tensor)

    # reporter = LungRADSReporter()
    # report = reporter.generate_report(case_result)

    # visualizer = ModelVisualizer()
    # dashboard = visualizer.create_dashboard(case_result)
