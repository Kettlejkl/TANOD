# app/api/video/feature_extractor.py
import torch
from torchvision import models, transforms
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleFeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, img_rgb):
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.squeeze()
            features = features / (torch.norm(features) + 1e-6)
        return features

feature_extractor = SimpleFeatureExtractor()
print(f"[INFO] Feature extractor loaded on GPU: {torch.cuda.get_device_name(0)} ✅")

def extract_feature(frame, box):
    """Extract ReID features from a bounding box
    frame: BGR image (numpy)
    box: (x, y, w, h)
    returns: torch tensor feature or None (if crop invalid)
    """
    x, y, w, h = box
    x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
    
    h_frame, w_frame = frame.shape[:2]
    x2, y2 = min(x + w, w_frame), min(y + h, h_frame)
    
    cropped = frame[y:y2, x:x2]
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return None
    
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return feature_extractor.extract(img_rgb)