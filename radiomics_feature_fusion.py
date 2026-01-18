
"""
Radiomics Feature Extraction and Multi-modal Fusion
This module implements radiomics feature extraction using PyRadiomics
and fusion with deep learning features and clinical data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk
from radiomics import featureextractor
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class RadiomicsExtractor:
    """Radiomics feature extractor using PyRadiomics"""

    def __init__(self, settings_file=None):
        """
        Initialize radiomics extractor

        Args:
            settings_file (str): Path to PyRadiomics settings file
        """
        if settings_file and os.path.exists(settings_file):
            self.extractor = featureextractor.RadiomicsFeatureExtractor(settings_file)
        else:
            # Use default settings
            self.extractor = featureextractor.RadiomicsFeatureExtractor()

            # Configure extraction settings
            self.extractor.enableImageTypeByName('Original')
            self.extractor.enableImageTypeByName('Wavelet')
            self.extractor.enableImageTypeByName('LoG')

            # Enable all feature classes
            self.extractor.enableFeatureClassByName('firstorder')
            self.extractor.enableFeatureClassByName('shape')
            self.extractor.enableFeatureClassByName('glcm')
            self.extractor.enableFeatureClassByName('glrlm')
            self.extractor.enableFeatureClassByName('glszm')
            self.extractor.enableFeatureClassByName('gldm')
            self.extractor.enableFeatureClassByName('ngtdm')

            # Set interpolation settings
            self.extractor.settings['interpolator'] = sitk.sitkBSpline
            self.extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
            self.extractor.settings['binWidth'] = 25

    def extract_features_from_arrays(self, image_array, mask_array):
        """
        Extract radiomics features from numpy arrays

        Args:
            image_array (np.ndarray): 3D image array
            mask_array (np.ndarray): 3D mask array

        Returns:
            dict: Extracted features
        """
        # Convert numpy arrays to SimpleITK images
        image = sitk.GetImageFromArray(image_array.astype(np.float32))
        mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

        # Extract features
        try:
            features = self.extractor.execute(image, mask)
            return dict(features)
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return {}

    def extract_features_from_files(self, image_path, mask_path):
        """
        Extract radiomics features from image files

        Args:
            image_path (str): Path to image file
            mask_path (str): Path to mask file

        Returns:
            dict: Extracted features
        """
        try:
            features = self.extractor.execute(image_path, mask_path)
            return dict(features)
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return {}

    def batch_extract_features(self, image_mask_pairs):
        """
        Extract features from multiple image-mask pairs

        Args:
            image_mask_pairs (list): List of (image_array, mask_array) tuples

        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        all_features = []

        for i, (image_array, mask_array) in enumerate(image_mask_pairs):
            print(f"Extracting features from sample {i+1}/{len(image_mask_pairs)}")
            features = self.extract_features_from_arrays(image_array, mask_array)

            if features:
                # Remove metadata and keep only numeric features
                numeric_features = {k: v for k, v in features.items() 
                                  if isinstance(v, (int, float)) and not k.startswith('diagnostics_')}
                numeric_features['sample_id'] = i
                all_features.append(numeric_features)

        return pd.DataFrame(all_features)

class ClinicalDataProcessor:
    """Processor for clinical data"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def preprocess_clinical_data(self, clinical_df, target_column=None):
        """
        Preprocess clinical data

        Args:
            clinical_df (pd.DataFrame): Clinical data
            target_column (str): Target column name (for training)

        Returns:
            np.ndarray: Processed clinical features
        """
        df = clinical_df.copy()

        # Remove target column if present
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])

        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].fillna('unknown'))
            else:
                # Handle unseen categories
                df[col] = df[col].fillna('unknown')
                unique_values = set(df[col].unique())
                known_values = set(self.label_encoders[col].classes_)

                # Map unknown values to a default value
                for val in unique_values - known_values:
                    df[col] = df[col].replace(val, 'unknown')

                df[col] = self.label_encoders[col].transform(df[col])

        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        # Scale features
        if not self.fitted:
            scaled_features = self.scaler.fit_transform(df)
            self.fitted = True
        else:
            scaled_features = self.scaler.transform(df)

        return scaled_features

class FeatureFusionNetwork(nn.Module):
    """Neural network for fusing radiomics, deep learning, and clinical features"""

    def __init__(self, radiomics_dim, deep_features_dim, clinical_dim, 
                 output_dim=1, hidden_dims=[512, 256, 128]):
        super(FeatureFusionNetwork, self).__init__()

        self.radiomics_dim = radiomics_dim
        self.deep_features_dim = deep_features_dim
        self.clinical_dim = clinical_dim

        # Radiomics branch
        self.radiomics_branch = nn.Sequential(
            nn.Linear(radiomics_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[1] // 2),
            nn.BatchNorm1d(hidden_dims[1] // 2),
            nn.ReLU(inplace=True)
        )

        # Deep features branch
        self.deep_features_branch = nn.Sequential(
            nn.Linear(deep_features_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[1] // 2),
            nn.BatchNorm1d(hidden_dims[1] // 2),
            nn.ReLU(inplace=True)
        )

        # Clinical data branch
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dims[2] // 2),
            nn.BatchNorm1d(hidden_dims[2] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Fusion layers
        fusion_input_dim = hidden_dims[1] + hidden_dims[2] // 2
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.BatchNorm1d(hidden_dims[2] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[2] // 2, output_dim)
        )

        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_input_dim // 4, fusion_input_dim),
            nn.Sigmoid()
        )

    def forward(self, radiomics_features, deep_features, clinical_features):
        # Process each modality
        radiomics_out = self.radiomics_branch(radiomics_features)
        deep_out = self.deep_features_branch(deep_features)
        clinical_out = self.clinical_branch(clinical_features)

        # Concatenate features
        fused_features = torch.cat([radiomics_out, deep_out, clinical_out], dim=1)

        # Apply attention
        attention_weights = self.attention(fused_features)
        attended_features = fused_features * attention_weights

        # Final prediction
        output = self.fusion_layers(attended_features)

        return output, attention_weights

class MetaClassifier:
    """Meta-classifier using traditional ML methods"""

    def __init__(self, classifier_type='xgboost'):
        self.classifier_type = classifier_type
        self.classifier = None
        self.feature_selector = None

    def train(self, features, labels, n_features=100):
        """
        Train meta-classifier

        Args:
            features (np.ndarray): Input features
            labels (np.ndarray): Target labels
            n_features (int): Number of top features to select
        """
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(n_features, features.shape[1]))
        selected_features = self.feature_selector.fit_transform(features, labels)

        # Train classifier
        if self.classifier_type == 'xgboost':
            import xgboost as xgb
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.classifier_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.classifier_type == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(random_state=42)

        self.classifier.fit(selected_features, labels)

    def predict(self, features):
        """Predict using trained classifier"""
        if self.classifier is None or self.feature_selector is None:
            raise ValueError("Classifier not trained yet!")

        selected_features = self.feature_selector.transform(features)
        return self.classifier.predict_proba(selected_features)[:, 1]

    def get_feature_importance(self):
        """Get feature importance scores"""
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            return abs(self.classifier.coef_[0])
        else:
            return None

class MultiModalPipeline:
    """Complete multi-modal pipeline for lung cancer prediction"""

    def __init__(self):
        self.radiomics_extractor = RadiomicsExtractor()
        self.clinical_processor = ClinicalDataProcessor()
        self.fusion_network = None
        self.meta_classifier = MetaClassifier()

    def extract_all_features(self, image_patches, masks, clinical_data, deep_features):
        """
        Extract and process all types of features

        Args:
            image_patches (list): List of 3D image patches
            masks (list): List of corresponding masks
            clinical_data (pd.DataFrame): Clinical data
            deep_features (np.ndarray): Deep learning features

        Returns:
            dict: All processed features
        """
        # Extract radiomics features
        print("Extracting radiomics features...")
        image_mask_pairs = list(zip(image_patches, masks))
        radiomics_df = self.radiomics_extractor.batch_extract_features(image_mask_pairs)

        # Process clinical data
        print("Processing clinical data...")
        clinical_features = self.clinical_processor.preprocess_clinical_data(clinical_data)

        return {
            'radiomics': radiomics_df.values[:, :-1],  # Exclude sample_id
            'clinical': clinical_features,
            'deep_features': deep_features
        }

    def train_fusion_network(self, features_dict, labels, num_epochs=100, device='cuda'):
        """Train the feature fusion network"""

        # Initialize network
        radiomics_dim = features_dict['radiomics'].shape[1]
        deep_features_dim = features_dict['deep_features'].shape[1]
        clinical_dim = features_dict['clinical'].shape[1]

        self.fusion_network = FeatureFusionNetwork(
            radiomics_dim=radiomics_dim,
            deep_features_dim=deep_features_dim,
            clinical_dim=clinical_dim,
            output_dim=1
        )

        # Training setup
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fusion_network.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.fusion_network.parameters(), lr=1e-3, weight_decay=1e-4)

        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(features_dict['radiomics']).to(device)
        deep_features_tensor = torch.FloatTensor(features_dict['deep_features']).to(device)
        clinical_tensor = torch.FloatTensor(features_dict['clinical']).to(device)
        labels_tensor = torch.FloatTensor(labels).to(device)

        # Training loop
        self.fusion_network.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs, attention = self.fusion_network(
                radiomics_tensor, deep_features_tensor, clinical_tensor
            )
            loss = criterion(outputs.squeeze(), labels_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def train_meta_classifier(self, features_dict, labels):
        """Train meta-classifier on all features"""

        # Combine all features
        all_features = np.concatenate([
            features_dict['radiomics'],
            features_dict['deep_features'],
            features_dict['clinical']
        ], axis=1)

        # Train meta-classifier
        self.meta_classifier.train(all_features, labels)

    def predict(self, image_patches, masks, clinical_data, deep_features):
        """Make predictions using the complete pipeline"""

        # Extract features
        features_dict = self.extract_all_features(
            image_patches, masks, clinical_data, deep_features
        )

        predictions = {}

        # Fusion network prediction
        if self.fusion_network is not None:
            self.fusion_network.eval()
            with torch.no_grad():
                radiomics_tensor = torch.FloatTensor(features_dict['radiomics'])
                deep_features_tensor = torch.FloatTensor(features_dict['deep_features'])
                clinical_tensor = torch.FloatTensor(features_dict['clinical'])

                fusion_output, attention = self.fusion_network(
                    radiomics_tensor, deep_features_tensor, clinical_tensor
                )
                predictions['fusion_network'] = torch.sigmoid(fusion_output).numpy()
                predictions['attention_weights'] = attention.numpy()

        # Meta-classifier prediction
        all_features = np.concatenate([
            features_dict['radiomics'],
            features_dict['deep_features'],
            features_dict['clinical']
        ], axis=1)
        predictions['meta_classifier'] = self.meta_classifier.predict(all_features)

        return predictions

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MultiModalPipeline()

    # Example usage with dummy data
    image_patches = [np.random.randn(32, 32, 32) for _ in range(10)]
    masks = [np.random.randint(0, 2, (32, 32, 32)) for _ in range(10)]
    clinical_data = pd.DataFrame({
        'age': np.random.randint(40, 80, 10),
        'gender': np.random.choice(['M', 'F'], 10),
        'smoking_history': np.random.choice(['Yes', 'No'], 10)
    })
    deep_features = np.random.randn(10, 512)
    labels = np.random.randint(0, 2, 10)

    # Extract features
    features_dict = pipeline.extract_all_features(
        image_patches, masks, clinical_data, deep_features
    )

    # Train models
    pipeline.train_fusion_network(features_dict, labels)
    pipeline.train_meta_classifier(features_dict, labels)

    # Make predictions
    predictions = pipeline.predict(image_patches, masks, clinical_data, deep_features)

    print("Pipeline training and prediction completed successfully!")
