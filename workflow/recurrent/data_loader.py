"""
Lazy data loader for memory-efficient feature loading.
Loads features on-demand instead of loading all features into memory at once.
"""

import numpy as np
import os
import json
from typing import Dict, Optional, Tuple


class LazyDataLoader:
    """
    Memory-efficient data loader that loads features on-demand.
    Supports both consolidated .npz files and separate feature files.
    """
    
    # Available feature types based on preprocessing notebook
    AVAILABLE_FEATURES = [
        'stft', 'mel_specs', 'mfccs', 'mctct', 'parakeet', 
        'seamlessM4T', 'whisper'
    ]
    
    def __init__(self, data_path: str, mode: str = 'auto'):
        """
        Initialize the lazy data loader.
        
        Args:
            data_path: Path to either:
                - A consolidated .npz file (legacy format)
                - A directory containing separate feature .npz files
            mode: 'auto' (detect), 'consolidated' (single file), or 'separate' (directory)
        """
        self.data_path = data_path
        self.mode = mode
        self._cache = {}  # Cache loaded features to avoid reloading
        self._labels = None
        self._labels_inorder = None
        self._class_count = None
        self._feature_shapes = {}  # Cache feature shapes for each feature type
        
        # Detect mode if auto
        if mode == 'auto':
            if os.path.isfile(data_path):
                self.mode = 'consolidated'
            elif os.path.isdir(data_path):
                self.mode = 'separate'
            else:
                raise ValueError(f"Data path {data_path} not found")
        
        # Load labels and metadata (these are small, load once)
        self._load_labels()
        self._load_metadata()
    
    def _load_labels(self):
        """Load labels (small, load once)"""
        if self.mode == 'consolidated':
            data = np.load(self.data_path, allow_pickle=True)
            self._labels = {
                'train': data['train_labels'],
                'val': data['val_labels'],
                'test': data['test_labels']
            }
            self._labels_inorder = data['labels_inorder']
            self._class_count = self._labels['train'].shape[1]
        else:
            # Try to load from separate labels file
            labels_path = os.path.join(self.data_path, 'labels.npz')
            if os.path.exists(labels_path):
                labels_data = np.load(labels_path, allow_pickle=True)
                self._labels = {
                    'train': labels_data['train_labels'],
                    'val': labels_data['val_labels'],
                    'test': labels_data['test_labels']
                }
                self._labels_inorder = list(labels_data['labels_inorder'])
                self._class_count = self._labels['train'].shape[1]
            else:
                # Fallback: try to infer from first available feature file
                for feat in self.AVAILABLE_FEATURES:
                    feat_path = os.path.join(self.data_path, f'{feat}_features.npz')
                    if os.path.exists(feat_path):
                        # Load just to get labels (will be same for all features)
                        temp_data = np.load(feat_path, allow_pickle=True)
                        # Labels might be in metadata or separate file
                        # For now, we'll need them passed separately
                        break
    
    def _load_metadata(self):
        """Load metadata if available"""
        if self.mode == 'separate':
            metadata_path = os.path.join(self.data_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
                    if 'languages' in self._metadata:
                        self._labels_inorder = self._metadata['languages']
    
    def get_feature_shape(self, feature_type: str) -> Tuple:
        """
        Get the feature shape for a given feature type without loading the data.
        
        Args:
            feature_type: Name of the feature type (e.g., 'stft', 'mel_specs')
        
        Returns:
            Tuple representing the feature shape (excluding batch dimension)
        """
        if feature_type in self._feature_shapes:
            return self._feature_shapes[feature_type]
        
        # Load just the first sample to get shape
        if self.mode == 'consolidated':
            data = np.load(self.data_path, allow_pickle=True)
            key = f'{feature_type}_train_features'
            if key in data:
                shape = data[key].shape[1:]  # Exclude batch dimension
                self._feature_shapes[feature_type] = shape
                return shape
        else:
            feat_path = os.path.join(self.data_path, f'{feature_type}_features.npz')
            if os.path.exists(feat_path):
                data = np.load(feat_path, allow_pickle=True)
                if 'train_features' in data:
                    shape = data['train_features'].shape[1:]
                    self._feature_shapes[feature_type] = shape
                    return shape
        
        raise ValueError(f"Feature type {feature_type} not found")
    
    def get_features(self, feature_type: str, split: str = 'train') -> np.ndarray:
        """
        Load features for a specific feature type and split.
        Uses caching to avoid reloading.
        
        Args:
            feature_type: Name of the feature type (e.g., 'stft', 'mel_specs')
            split: 'train', 'val', or 'test'
        
        Returns:
            numpy array of features
        """
        cache_key = f'{feature_type}_{split}'
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.mode == 'consolidated':
            data = np.load(self.data_path, allow_pickle=True)
            key = f'{feature_type}_{split}_features'
            if key not in data:
                raise ValueError(f"Feature {feature_type} not found in consolidated file")
            features = data[key]
        else:
            feat_path = os.path.join(self.data_path, f'{feature_type}_features.npz')
            if not os.path.exists(feat_path):
                raise ValueError(f"Feature file {feat_path} not found")
            
            data = np.load(feat_path, allow_pickle=True)
            key = f'{split}_features'
            if key not in data:
                raise ValueError(f"Split {split} not found in {feat_path}")
            features = data[key]
        
        # Cache it
        self._cache[cache_key] = features
        
        # Store shape if not already stored
        if feature_type not in self._feature_shapes:
            self._feature_shapes[feature_type] = features.shape[1:]
        
        return features
    
    def get_labels(self, split: str = 'train') -> np.ndarray:
        """Get labels for a specific split"""
        if self._labels is None:
            raise ValueError("Labels not loaded")
        return self._labels[split]
    
    @property
    def labels_inorder(self):
        """Get label order"""
        return self._labels_inorder
    
    @property
    def class_count(self):
        """Get number of classes"""
        return self._class_count
    
    def clear_cache(self, feature_type: Optional[str] = None):
        """
        Clear feature cache to free memory.
        
        Args:
            feature_type: If provided, clear only this feature type. Otherwise clear all.
        """
        if feature_type:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f'{feature_type}_')]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
    
    def get_data_dict(self, feature_type: str) -> Dict:
        """
        Get a data dictionary compatible with the existing Solution.develop_model interface.
        This loads the requested feature type and returns it in the expected format.
        
        Args:
            feature_type: Name of the feature type to load
        
        Returns:
            Dictionary with keys like '{feature_type}_train_features', etc.
        """
        return {
            f'{feature_type}_train_features': self.get_features(feature_type, 'train'),
            f'{feature_type}_val_features': self.get_features(feature_type, 'val'),
            f'{feature_type}_test_features': self.get_features(feature_type, 'test'),
            'train_labels': self.get_labels('train'),
            'val_labels': self.get_labels('val'),
            'test_labels': self.get_labels('test'),
            'labels_inorder': self.labels_inorder
        }
    
    def is_feature_available(self, feature_type: str) -> bool:
        """Check if a feature type is available"""
        try:
            self.get_feature_shape(feature_type)
            return True
        except (ValueError, KeyError, FileNotFoundError):
            return False

