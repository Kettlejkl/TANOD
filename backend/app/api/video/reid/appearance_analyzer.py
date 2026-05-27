"""
Appearance Analysis Module
Handles color histograms and clothing attribute extraction
"""

import cv2
import numpy as np
import logging


class AppearanceAnalyzer:
    """Extracts and compares color and clothing attributes"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('AppearanceAnalyzer')
        self.logger.setLevel(log_level)
    
    def extract_color_histogram(self, image_crop):
        """Extract color histogram from upper and lower body regions"""
        if image_crop is None or image_crop.size == 0:
            return None
        
        try:
            if len(image_crop.shape) != 3 or image_crop.shape[2] != 3:
                return None
            
            if image_crop.dtype != np.uint8:
                image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
            
            height, width = image_crop.shape[:2]
            upper_body = image_crop[:int(height*0.6), :]
            lower_body = image_crop[int(height*0.4):, :]
            
            hsv_upper = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            hsv_lower = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
            
            hist_upper_h = cv2.calcHist([hsv_upper], [0], None, [16], [0, 180])
            hist_upper_s = cv2.calcHist([hsv_upper], [1], None, [16], [0, 256])
            hist_lower_h = cv2.calcHist([hsv_lower], [0], None, [16], [0, 180])
            hist_lower_s = cv2.calcHist([hsv_lower], [1], None, [16], [0, 256])
            
            return {
                'upper_hue': cv2.normalize(hist_upper_h, hist_upper_h).flatten(),
                'upper_sat': cv2.normalize(hist_upper_s, hist_upper_s).flatten(),
                'lower_hue': cv2.normalize(hist_lower_h, hist_lower_h).flatten(),
                'lower_sat': cv2.normalize(hist_lower_s, hist_lower_s).flatten(),
            }
        except Exception as e:
            self.logger.error(f"Color extraction failed: {e}")
            return None
    
    def compare_color_histograms(self, color1, color2):
        """Compare two color histograms"""
        if color1 is None or color2 is None:
            return 0.0
        
        try:
            sim_upper_h = cv2.compareHist(color1['upper_hue'], color2['upper_hue'], 
                                          cv2.HISTCMP_CORREL)
            sim_upper_s = cv2.compareHist(color1['upper_sat'], color2['upper_sat'], 
                                          cv2.HISTCMP_CORREL)
            sim_lower_h = cv2.compareHist(color1['lower_hue'], color2['lower_hue'], 
                                          cv2.HISTCMP_CORREL)
            sim_lower_s = cv2.compareHist(color1['lower_sat'], color2['lower_sat'], 
                                          cv2.HISTCMP_CORREL)
            
            upper_score = 0.6 * sim_upper_h + 0.4 * sim_upper_s
            lower_score = 0.6 * sim_lower_h + 0.4 * sim_lower_s
            
            return max(0.0, min(1.0, 0.6 * upper_score + 0.4 * lower_score))
        except:
            return 0.0
    
    def extract_clothing_attributes(self, image_crop):
        """Extract dominant colors from upper and lower body clothing"""
        if image_crop is None or image_crop.size == 0:
            return None
        
        try:
            height, width = image_crop.shape[:2]
            upper = image_crop[:int(height*0.5), :]
            lower = image_crop[int(height*0.5):, :]
            
            upper_colors = self._get_dominant_colors(upper, n=3)
            lower_colors = self._get_dominant_colors(lower, n=2)
            
            return {
                'upper_colors': upper_colors,
                'lower_colors': lower_colors,
            }
        except:
            return None
    
    def _get_dominant_colors(self, image, n=3):
        """Extract n dominant colors using k-means clustering"""
        try:
            pixels = image.reshape(-1, 3).astype(np.float32)
            if len(pixels) == 0:
                return []
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, n, None, criteria, 10, 
                                           cv2.KMEANS_RANDOM_CENTERS)
            
            centers = centers.astype(np.uint8)
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            
            return [tuple(centers[i]) for i in sorted_indices[:n]]
        except:
            return []
    
    def compare_clothing_attributes(self, attr1, attr2):
        """Compare clothing attributes between two detections"""
        if attr1 is None or attr2 is None:
            return 0.0
        
        try:
            score = 0.0
            weight_sum = 0.0
            
            if attr1.get('upper_colors') and attr2.get('upper_colors'):
                upper_sim = self._compare_color_lists(attr1['upper_colors'], 
                                                      attr2['upper_colors'])
                score += upper_sim * 0.6
                weight_sum += 0.6
            
            if attr1.get('lower_colors') and attr2.get('lower_colors'):
                lower_sim = self._compare_color_lists(attr1['lower_colors'], 
                                                      attr2['lower_colors'])
                score += lower_sim * 0.4
                weight_sum += 0.4
            
            return score / weight_sum if weight_sum > 0 else 0.0
        except:
            return 0.0
    
    def _compare_color_lists(self, colors1, colors2):
        """Compare two lists of RGB colors"""
        if not colors1 or not colors2:
            return 0.0
        
        max_sim = 0.0
        for c1 in colors1:
            for c2 in colors2:
                sim = 1.0 - (np.linalg.norm(np.array(c1) - np.array(c2)) / 441.67)
                max_sim = max(max_sim, sim)
        
        return max_sim