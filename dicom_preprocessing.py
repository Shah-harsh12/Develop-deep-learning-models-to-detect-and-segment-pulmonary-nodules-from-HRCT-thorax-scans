"""
DICOM Preprocessing Module for LUNA16 Dataset
This module handles DICOM file loading, de-identification, quality control,
and preprocessing for the lung cancer detection pipeline including lung mask generation.
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from scipy import ndimage
from skimage import measure, segmentation
import warnings
warnings.filterwarnings('ignore')

class DICOMPreprocessor:
    def __init__(self, data_dir, output_dir, target_spacing=(1.0, 1.0, 1.0)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_spacing = target_spacing

        # HU windowing parameters
        self.lung_window = (-1200, 600)  # For lung tissue

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'masks'), exist_ok=True)

    def load_mhd_image(self, mhd_path):
        image = sitk.ReadImage(mhd_path)
        return image

    def resample_image(self, image, target_spacing):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(np.round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkBSpline)

        return resampler.Execute(image)

    def apply_hu_windowing(self, image_array, window):
        min_hu, max_hu = window
        image_array = np.clip(image_array, min_hu, max_hu)
        return image_array

    def normalize_hu(self, image_array, window):
        min_hu, max_hu = window
        image_array = (image_array - min_hu) / (max_hu - min_hu)
        image_array = np.clip(image_array, 0, 1)
        return image_array

    def denoise_image(self, image_array, sigma=1.0):
        return ndimage.gaussian_filter(image_array, sigma=sigma)

    def zero_center_normalize(self, image_array):
        mean = np.mean(image_array)
        std = np.std(image_array)
        return (image_array - mean) / std

import numpy as np
from scipy import ndimage

def generate_lung_mask_alternative(image_array):
    """
    Alternative lung mask generation for CT image volume.

    Steps:
    - Threshold below -400 HU for lung air/tissue separation
    - Remove air pockets connected to image borders
    - Keep largest two connected components (lungs)
    - Fill holes inside these components
    - Apply morphological closing to close gaps

    Args:
        image_array (np.ndarray): 3D CT image array in Hounsfield units (HU)

    Returns:
        np.ndarray: Binary lung mask (1: lung, 0: background)
    """
    # Threshold image to segment lung region
    binary_img = image_array < -400

    # Remove structures connected to border (air outside body)
    cleared = segmentation.clear_border(binary_img)

    # Label connected components
    labeled_img, num_features = ndimage.label(cleared)

    if num_features == 0:
        return np.zeros_like(binary_img, dtype=np.uint8)

    # Get sizes of all components
    sizes = ndimage.sum(cleared, labeled_img, range(1, num_features + 1))

    # Select two largest components (lungs)
    if len(sizes) > 1:
        largest_indices = np.argsort(sizes)[-2:] + 1
        lung_mask = np.zeros_like(binary_img, dtype=np.uint8)
        for i in largest_indices:
            lung_mask[labeled_img == i] = 1
    else:
        # Only one component found, use it
        lung_mask = (labeled_img == 1).astype(np.uint8)

    # Fill holes inside lungs
    lung_mask = ndimage.binary_fill_holes(lung_mask).astype(np.uint8)

    # Morphological closing (3x3x3) to close gaps
    struct = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_closing(lung_mask, structure=struct, iterations=2).astype(np.uint8)

    return lung_mask

    def process_single_case(self, case_id, mhd_path):
        try:
            image = self.load_mhd_image(mhd_path)
            resampled_image = self.resample_image(image, self.target_spacing)
            image_array = sitk.GetArrayFromImage(resampled_image)

            # Apply lung windowing
            windowed_array = self.apply_hu_windowing(image_array, self.lung_window)

            # Normalize to 0-1
            normalized_array = self.normalize_hu(windowed_array, self.lung_window)

            # Denoise
            denoised_array = self.denoise_image(normalized_array, sigma=0.5)

            # Generate lung mask from original HU image array (resampled before windowing)
            lung_mask = self.generate_lung_mask(image_array)

            # Save processed image
            clean_path = os.path.join(self.output_dir, 'images', f'{case_id}_clean.npy')
            np.save(clean_path, denoised_array.astype(np.float32))

            # Save lung mask
            mask_path = os.path.join(self.output_dir, 'masks', f'{case_id}_mask.npy')
            np.save(mask_path, lung_mask.astype(np.uint8))

            metadata = {
                'case_id': case_id,
                'original_spacing': image.GetSpacing(),
                'original_size': image.GetSize(),
                'processed_spacing': self.target_spacing,
                'processed_shape': denoised_array.shape,
                'origin': resampled_image.GetOrigin(),
                'direction': resampled_image.GetDirection()
            }

            return metadata

        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            return None

    def process_luna16_dataset(self, annotations_csv):
        annotations = pd.read_csv(annotations_csv)
        case_ids = annotations['seriesuid'].unique()

        processed_metadata = []

        for case_id in case_ids:
            mhd_path = os.path.join(self.data_dir, f'{case_id}.mhd')

            if os.path.exists(mhd_path):
                print(f"Processing case: {case_id}")
                metadata = self.process_single_case(case_id, mhd_path)
                if metadata:
                    processed_metadata.append(metadata)
            else:
                print(f"Warning: File not found - {mhd_path}")

        metadata_df = pd.DataFrame(processed_metadata)
        metadata_df.to_csv(os.path.join(self.output_dir, 'processing_metadata.csv'), index=False)

        print(f"Processing complete. Processed {len(processed_metadata)} cases.")
        return processed_metadata