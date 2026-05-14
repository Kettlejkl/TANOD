"""
Feature Extraction Module - Torchreid Implementation
Handles all ReID feature extraction and preprocessing using Torchreid v1.4.0
"""

import torch
import torchreid
import numpy as np
from PIL import Image
from torchvision import transforms as T
import logging


class FeatureExtractor:
    """Extracts and normalizes ReID features from person crops using Torchreid"""
    
    def __init__(self, model_name='osnet_ibn_x1_0', use_gpu=True, log_level=logging.INFO):
        """
        Initialize Torchreid feature extractor
        
        Args:
            model_name: Model architecture (osnet_ibn_x1_0 recommended for surveillance)
            use_gpu: Whether to use GPU
            log_level: Logging level
        """
        self.logger = logging.getLogger('FeatureExtractor')
        self.logger.setLevel(log_level)
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading model: {model_name} on {self.device}")
        
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.feature_dim = 512
        self.model_name = model_name
        
        self.logger.info(f"Feature extractor initialized (dim={self.feature_dim})")
    
    def extract(self, image_crop):
        """Extract normalized feature vector from image crop"""
        try:
            if image_crop is None or image_crop.size == 0:
                return None
            
            if image_crop.shape[0] < 64 or image_crop.shape[1] < 32:
                self.logger.warning(f"Crop too small: {image_crop.shape}")
                return None
            
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                if image_crop.dtype != np.uint8:
                    image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
                image_rgb = Image.fromarray(image_crop[:, :, ::-1])
            else:
                if image_crop.dtype != np.uint8:
                    image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
                image_rgb = Image.fromarray(image_crop)
            
            if image_rgb.mode != 'RGB':
                image_rgb = image_rgb.convert('RGB')
            
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            features = features.cpu().numpy().flatten()
            return self.normalize(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    @staticmethod
    def normalize(feature):
        """Normalize feature vector to unit length"""
        if feature is None:
            return None
        
        if isinstance(feature, np.ndarray):
            feat_np = feature
        elif hasattr(feature, 'is_cuda') and feature.is_cuda:
            feat_np = feature.detach().cpu().numpy()
        elif hasattr(feature, 'numpy'):
            feat_np = feature.numpy()
        else:
            feat_np = np.array(feature)
        
        feat_np = feat_np.flatten()
        norm = np.linalg.norm(feat_np)
        return feat_np / norm if norm > 0 else feat_np
    
    def extract_batch(self, image_crops, batch_size=32):
        """
        Extract features from multiple crops efficiently
        
        Args:
            image_crops: List of image crops (numpy arrays)
            batch_size: Batch size for processing
            
        Returns:
            List of feature vectors
        """
        if not image_crops:
            return []
        
        features = []
        
        for i in range(0, len(image_crops), batch_size):
            batch = image_crops[i:i + batch_size]
            batch_tensors = []
            
            for crop in batch:
                try:
                    if crop is None or crop.size == 0:
                        features.append(None)
                        continue
                    
                    if crop.shape[0] < 64 or crop.shape[1] < 32:
                        features.append(None)
                        continue
                    
                    if len(crop.shape) == 3 and crop.shape[2] == 3:
                        if crop.dtype != np.uint8:
                            crop = np.clip(crop, 0, 255).astype(np.uint8)
                        img_rgb = Image.fromarray(crop[:, :, ::-1])
                    else:
                        if crop.dtype != np.uint8:
                            crop = np.clip(crop, 0, 255).astype(np.uint8)
                        img_rgb = Image.fromarray(crop)
                    
                    if img_rgb.mode != 'RGB':
                        img_rgb = img_rgb.convert('RGB')
                    
                    img_tensor = self.transform(img_rgb)
                    batch_tensors.append(img_tensor)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process crop: {e}")
                    features.append(None)
            
            if not batch_tensors:
                continue
            
            try:
                batch_input = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    batch_features = self.model(batch_input)
                
                batch_features = batch_features.cpu().numpy()
                
                for feat in batch_features:
                    features.append(self.normalize(feat))
                    
            except Exception as e:
                self.logger.error(f"Batch extraction failed: {e}")
                features.extend([None] * len(batch_tensors))
        
        return features
    
    def get_feature_dim(self):
        """Get feature vector dimension"""
        return self.feature_dim
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': self.model_name,
            'feature_dim': self.feature_dim,
            'device': str(self.device)
        }


AVAILABLE_MODELS = {
    'osnet_ibn_x1_0': 'OSNet with IBN - Best for surveillance (512-dim)',
    'osnet_x1_0': 'OSNet 1.0x - Good balance (512-dim)',
    'osnet_x0_75': 'OSNet 0.75x - Faster (512-dim)',
    'osnet_x0_5': 'OSNet 0.5x - Fastest (512-dim)',
    'osnet_x0_25': 'OSNet 0.25x - Lightest (512-dim)',
    'resnet50': 'ResNet50 - Classic (2048-dim)',
    'resnet101': 'ResNet101 - More capacity (2048-dim)',
    'densenet121': 'DenseNet121 - Dense connections',
    'mobilenetv2_x1_0': 'MobileNetV2 - Mobile devices',
    'shufflenet': 'ShuffleNet - Efficient'
}