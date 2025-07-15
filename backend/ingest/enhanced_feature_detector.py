"""
Enhanced Feature Detector for video keyframes.
Extracts multiple types of features for richer semantic understanding.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Import detection models
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Detected object in a keyframe."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: float

@dataclass
class SceneFeatures:
    """Scene classification features."""
    scene_type: str
    indoor_outdoor: str
    lighting: str
    time_of_day: str
    confidence: float

@dataclass
class ActionFeatures:
    """Action/activity recognition features."""
    primary_action: str
    secondary_actions: List[str]
    confidence: float
    temporal_context: List[str]

@dataclass
class TextFeatures:
    """Text detection and OCR features."""
    detected_text: List[str]
    text_regions: List[Tuple[int, int, int, int]]
    language: str
    confidence: float

@dataclass
class AudioFeatures:
    """Audio analysis features (for shot-level analysis)."""
    audio_type: str  # speech, music, silence, noise
    has_speech: bool
    music_detected: bool
    volume_level: float
    transcription: Optional[str] = None

@dataclass
class EnhancedFeatures:
    """Complete feature set for a keyframe."""
    keyframe_id: str
    shot_id: str
    timestamp: float
    
    # Visual features
    objects: List[DetectedObject]
    scene: SceneFeatures
    actions: ActionFeatures
    text: TextFeatures
    
    # Derived features for search
    searchable_tags: List[str]
    dominant_colors: List[str]
    composition_type: str  # portrait, landscape, close-up, wide
    
    # Feature vectors
    visual_features: np.ndarray
    semantic_features: np.ndarray
    
    # Metadata
    extraction_time: float
    model_versions: Dict[str, str]

class EnhancedFeatureDetector:
    """
    Comprehensive feature detector for video keyframes.
    Combines multiple detection models for rich semantic understanding.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 device: str = "auto",
                 batch_size: int = 4):
        """
        Initialize the feature detector.
        
        Args:
            models_dir: Directory containing model files
            device: Device to use (cuda, cpu, or auto)
            batch_size: Batch size for processing
        """
        self.models_dir = Path(models_dir)
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.models = {}
        self.model_versions = {}
        
        # Initialize detection models
        asyncio.create_task(self._initialize_models())
        
        # Scene classification mappings
        self.scene_mappings = self._load_scene_mappings()
        
        # Action classification mappings
        self.action_mappings = self._load_action_mappings()
    
    async def _initialize_models(self):
        """Initialize all detection models."""
        logger.info("Initializing enhanced feature detection models...")
        
        try:
            # Object Detection - YOLO
            if YOLO_AVAILABLE:
                await self._load_yolo_model()
            
            # Scene Classification - ResNet
            if TORCH_AVAILABLE:
                await self._load_scene_model()
            
            # OCR - EasyOCR
            if OCR_AVAILABLE:
                await self._load_ocr_model()
            
            # Action Recognition
            await self._load_action_model()
            
            logger.info("Feature detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load some models: {e}")
    
    async def _load_yolo_model(self):
        """Load YOLO object detection model."""
        try:
            # Use YOLOv8 nano for speed
            self.models['yolo'] = YOLO('yolov8n.pt')
            self.model_versions['yolo'] = 'yolov8n'
            logger.info("YOLO object detection model loaded")
        except Exception as e:
            logger.warning(f"YOLO model not available: {e}")
    
    async def _load_scene_model(self):
        """Load scene classification model."""
        try:
            # Use pre-trained ResNet for scene classification
            model = resnet50(pretrained=True)
            model.eval()
            model = model.to(self.device)
            self.models['scene'] = model
            self.model_versions['scene'] = 'resnet50_places365'
            
            # Image preprocessing
            self.scene_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Scene classification model loaded")
        except Exception as e:
            logger.warning(f"Scene model not available: {e}")
    
    async def _load_ocr_model(self):
        """Load OCR model."""
        try:
            self.models['ocr'] = easyocr.Reader(['en'], gpu=self.device.type == 'cuda')
            self.model_versions['ocr'] = 'easyocr_en'
            logger.info("OCR model loaded")
        except Exception as e:
            logger.warning(f"OCR model not available: {e}")
    
    async def _load_action_model(self):
        """Load action recognition model (placeholder)."""
        try:
            # For now, use rule-based action detection
            # In production, could use video action recognition models
            self.models['action'] = self._rule_based_action_detector
            self.model_versions['action'] = 'rule_based_v1'
            logger.info("Action recognition model loaded")
        except Exception as e:
            logger.warning(f"Action model not available: {e}")
    
    def _load_scene_mappings(self) -> Dict[str, Any]:
        """Load scene classification mappings."""
        return {
            "indoor_keywords": ["room", "kitchen", "office", "bedroom", "bathroom", "hall"],
            "outdoor_keywords": ["park", "street", "beach", "mountain", "garden", "field"],
            "lighting_types": {
                "bright": [0.7, 1.0],
                "normal": [0.3, 0.7],
                "dark": [0.0, 0.3]
            },
            "time_indicators": {
                "morning": ["sunrise", "dawn", "early"],
                "day": ["sun", "bright", "daylight"],
                "evening": ["sunset", "dusk", "golden"],
                "night": ["dark", "lights", "moon", "stars"]
            }
        }
    
    def _load_action_mappings(self) -> Dict[str, Any]:
        """Load action recognition mappings."""
        return {
            "movement_actions": ["walking", "running", "jumping", "dancing", "swimming"],
            "object_interactions": ["cooking", "eating", "drinking", "reading", "writing"],
            "social_actions": ["talking", "meeting", "hugging", "shaking hands", "waving"],
            "sports_actions": ["playing", "kicking", "throwing", "catching", "hitting"],
            "work_actions": ["typing", "presenting", "building", "fixing", "cleaning"]
        }
    
    async def extract_features(self, 
                             keyframe_path: str, 
                             keyframe_id: str,
                             shot_id: str,
                             timestamp: float,
                             audio_features: Optional[AudioFeatures] = None) -> EnhancedFeatures:
        """
        Extract comprehensive features from a keyframe.
        
        Args:
            keyframe_path: Path to keyframe image
            keyframe_id: Unique keyframe identifier
            shot_id: Shot identifier
            timestamp: Timestamp in video
            audio_features: Optional audio features for the shot
            
        Returns:
            EnhancedFeatures object with all detected features
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(keyframe_path)
        if image is None:
            raise ValueError(f"Could not load image: {keyframe_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features in parallel
        tasks = []
        
        # Object detection
        if 'yolo' in self.models:
            tasks.append(self._detect_objects(image_rgb))
        else:
            tasks.append(asyncio.create_task(self._placeholder_objects()))
        
        # Scene classification
        if 'scene' in self.models:
            tasks.append(self._classify_scene(image_rgb))
        else:
            tasks.append(asyncio.create_task(self._placeholder_scene()))
        
        # Text detection
        if 'ocr' in self.models:
            tasks.append(self._detect_text(image_rgb))
        else:
            tasks.append(asyncio.create_task(self._placeholder_text()))
        
        # Action recognition
        if 'action' in self.models:
            tasks.append(self._recognize_actions(image_rgb))
        else:
            tasks.append(asyncio.create_task(self._placeholder_actions()))
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        objects, scene, text, actions = results
        
        # Extract additional visual features
        visual_features = await self._extract_visual_features(image_rgb)
        semantic_features = await self._extract_semantic_features(image_rgb, objects, scene)
        
        # Generate searchable tags
        searchable_tags = self._generate_searchable_tags(objects, scene, actions, text)
        
        # Analyze composition and colors
        composition_type = self._analyze_composition(image_rgb)
        dominant_colors = self._extract_dominant_colors(image_rgb)
        
        # Create enhanced features
        features = EnhancedFeatures(
            keyframe_id=keyframe_id,
            shot_id=shot_id,
            timestamp=timestamp,
            objects=objects,
            scene=scene,
            actions=actions,
            text=text,
            searchable_tags=searchable_tags,
            dominant_colors=dominant_colors,
            composition_type=composition_type,
            visual_features=visual_features,
            semantic_features=semantic_features,
            extraction_time=time.time() - start_time,
            model_versions=self.model_versions.copy()
        )
        
        return features
    
    async def _detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects using YOLO."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.models['yolo'],
                image
            )
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.models['yolo'].names[class_id]
                        
                        # Calculate center and area
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        area = (x2 - x1) * (y2 - y1)
                        
                        objects.append(DetectedObject(
                            name=class_name,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=center,
                            area=float(area)
                        ))
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    async def _classify_scene(self, image: np.ndarray) -> SceneFeatures:
        """Classify scene type and attributes."""
        try:
            # Basic scene analysis
            height, width = image.shape[:2]
            
            # Analyze lighting
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray) / 255.0
            
            if brightness > 0.7:
                lighting = "bright"
            elif brightness > 0.3:
                lighting = "normal"
            else:
                lighting = "dark"
            
            # Simple indoor/outdoor detection based on color distribution
            # This could be replaced with a proper scene classification model
            blue_ratio = np.mean(image[:, :, 2]) / 255.0  # Sky indicator
            green_ratio = np.mean(image[:, :, 1]) / 255.0  # Vegetation indicator
            
            if blue_ratio > 0.6 or green_ratio > 0.5:
                indoor_outdoor = "outdoor"
                scene_type = "outdoor_scene"
            else:
                indoor_outdoor = "indoor"
                scene_type = "indoor_scene"
            
            # Time of day estimation
            if brightness > 0.8:
                time_of_day = "day"
            elif brightness > 0.4:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            
            return SceneFeatures(
                scene_type=scene_type,
                indoor_outdoor=indoor_outdoor,
                lighting=lighting,
                time_of_day=time_of_day,
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return await self._placeholder_scene()
    
    async def _detect_text(self, image: np.ndarray) -> TextFeatures:
        """Detect and recognize text in image."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.models['ocr'].readtext,
                image
            )
            
            detected_text = []
            text_regions = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence detections
                    detected_text.append(text.strip())
                    
                    # Convert bbox to rectangle format
                    points = np.array(bbox, dtype=np.int32)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    text_regions.append((x1, y1, x2, y2))
            
            return TextFeatures(
                detected_text=detected_text,
                text_regions=text_regions,
                language="en",  # EasyOCR auto-detects, but we specified English
                confidence=0.8 if detected_text else 0.0
            )
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return await self._placeholder_text()
    
    async def _recognize_actions(self, image: np.ndarray) -> ActionFeatures:
        """Recognize actions and activities in the image."""
        try:
            # Rule-based action recognition (placeholder)
            # In production, this could use a proper action recognition model
            
            primary_action = "unknown"
            secondary_actions = []
            confidence = 0.5
            
            # Simple rule-based detection
            # This is a placeholder - real implementation would use video action models
            
            return ActionFeatures(
                primary_action=primary_action,
                secondary_actions=secondary_actions,
                confidence=confidence,
                temporal_context=[]
            )
            
        except Exception as e:
            logger.error(f"Action recognition failed: {e}")
            return await self._placeholder_actions()
    
    def _rule_based_action_detector(self, image: np.ndarray) -> ActionFeatures:
        """Simple rule-based action detection."""
        # This is a placeholder implementation
        return ActionFeatures(
            primary_action="unknown",
            secondary_actions=[],
            confidence=0.5,
            temporal_context=[]
        )
    
    async def _extract_visual_features(self, image: np.ndarray) -> np.ndarray:
        """Extract visual feature vector from image."""
        try:
            # Simple visual features (histogram, edges, etc.)
            # In production, could use deep features from CNN
            
            # Color histogram
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            # Edge features
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Texture features (simple)
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Combine features
            features = np.concatenate([
                hist_r.flatten(),
                hist_g.flatten(),
                hist_b.flatten(),
                [edge_density, texture]
            ])
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            return np.zeros(98, dtype=np.float32)  # 32*3 + 2
    
    async def _extract_semantic_features(self, 
                                       image: np.ndarray, 
                                       objects: List[DetectedObject],
                                       scene: SceneFeatures) -> np.ndarray:
        """Extract semantic feature vector."""
        try:
            # Create semantic feature vector from detected elements
            features = []
            
            # Object presence features (top 20 common objects)
            common_objects = ["person", "car", "dog", "cat", "chair", "table", "book", 
                            "phone", "laptop", "tv", "bed", "bottle", "cup", "fork",
                            "knife", "spoon", "bowl", "banana", "apple", "sandwich"]
            
            for obj_name in common_objects:
                has_object = any(obj.name == obj_name for obj in objects)
                features.append(1.0 if has_object else 0.0)
            
            # Scene features
            features.extend([
                1.0 if scene.indoor_outdoor == "indoor" else 0.0,
                1.0 if scene.indoor_outdoor == "outdoor" else 0.0,
                1.0 if scene.lighting == "bright" else 0.0,
                1.0 if scene.lighting == "normal" else 0.0,
                1.0 if scene.lighting == "dark" else 0.0
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Semantic feature extraction failed: {e}")
            return np.zeros(25, dtype=np.float32)  # 20 objects + 5 scene
    
    def _generate_searchable_tags(self, 
                                objects: List[DetectedObject],
                                scene: SceneFeatures,
                                actions: ActionFeatures,
                                text: TextFeatures) -> List[str]:
        """Generate searchable tags from all features."""
        tags = []
        
        # Object tags
        for obj in objects:
            if obj.confidence > 0.5:
                tags.append(obj.name)
        
        # Scene tags
        tags.extend([scene.scene_type, scene.indoor_outdoor, scene.lighting, scene.time_of_day])
        
        # Action tags
        if actions.primary_action != "unknown":
            tags.append(actions.primary_action)
        tags.extend(actions.secondary_actions)
        
        # Text tags
        tags.extend(text.detected_text)
        
        # Remove duplicates and filter
        tags = list(set([tag.lower().strip() for tag in tags if tag and len(tag) > 1]))
        
        return tags[:20]  # Limit to top 20 tags
    
    def _analyze_composition(self, image: np.ndarray) -> str:
        """Analyze image composition type."""
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            return "landscape"
        elif aspect_ratio < 0.8:
            return "portrait"
        else:
            return "square"
    
    def _extract_dominant_colors(self, image: np.ndarray) -> List[str]:
        """Extract dominant colors from image."""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = []
            for color in kmeans.cluster_centers_:
                # Convert to color name (simplified)
                r, g, b = color.astype(int)
                if r > 150 and g > 150 and b > 150:
                    colors.append("white")
                elif r < 50 and g < 50 and b < 50:
                    colors.append("black")
                elif r > g and r > b:
                    colors.append("red")
                elif g > r and g > b:
                    colors.append("green")
                elif b > r and b > g:
                    colors.append("blue")
                else:
                    colors.append("gray")
            
            return list(set(colors))
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return ["unknown"]
    
    # Placeholder methods for when models are not available
    async def _placeholder_objects(self) -> List[DetectedObject]:
        return []
    
    async def _placeholder_scene(self) -> SceneFeatures:
        return SceneFeatures(
            scene_type="unknown",
            indoor_outdoor="unknown",
            lighting="normal",
            time_of_day="day",
            confidence=0.0
        )
    
    async def _placeholder_text(self) -> TextFeatures:
        return TextFeatures(
            detected_text=[],
            text_regions=[],
            language="en",
            confidence=0.0
        )
    
    async def _placeholder_actions(self) -> ActionFeatures:
        return ActionFeatures(
            primary_action="unknown",
            secondary_actions=[],
            confidence=0.0,
            temporal_context=[]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature detector statistics."""
        return {
            "models_loaded": list(self.models.keys()),
            "model_versions": self.model_versions,
            "device": str(self.device),
            "available_features": {
                "object_detection": "yolo" in self.models,
                "scene_classification": "scene" in self.models,
                "text_detection": "ocr" in self.models,
                "action_recognition": "action" in self.models
            }
        }
