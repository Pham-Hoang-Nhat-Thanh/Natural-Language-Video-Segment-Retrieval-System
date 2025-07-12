import torch
import clip
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from PIL import Image
import cv2
import json
import time

from keyframe_extractor import Keyframe

logger = logging.getLogger(__name__)

@dataclass
class Embedding:
    keyframe_id: str
    embedding: np.ndarray
    dimension: int
    model_name: str
    creation_time: float

class EmbeddingService:
    """
    CLIP-based embedding service with ONNX optimization
    """
    
    def __init__(self, 
                 model_path: str = "models/clip",
                 model_name: str = "ViT-B/32",
                 device: str = "auto",
                 use_onnx: bool = True,
                 batch_size: int = 32):
        """
        Initialize embedding service
        
        Args:
            model_path: Path to model files
            model_name: CLIP model variant to use
            device: Device to run on ('auto', 'cpu', 'cuda')
            use_onnx: Whether to use ONNX optimized model
            batch_size: Batch size for processing
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_onnx = use_onnx
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Model components
        self.clip_model = None
        self.clip_preprocess = None
        self.onnx_session = None
        self.tokenizer = None
        self.embedding_dim = 512  # Default for ViT-B/32
        
        logger.info(f"Initialized EmbeddingService with model: {model_name}, device: {device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    async def load_model(self):
        """Load CLIP model and prepare for inference"""
        logger.info("Loading CLIP model...")
        
        try:
            if self.use_onnx and self._onnx_model_exists():
                await self._load_onnx_model()
            else:
                await self._load_pytorch_model()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _onnx_model_exists(self) -> bool:
        """Check if ONNX model files exist (new automated format)"""
        # Check for new automated ONNX format
        model_prefix = f"clip_{self.model_name.replace('/', '_')}"
        text_onnx_path = self.model_path / "onnx" / f"{model_prefix}_text_encoder.onnx"
        image_onnx_path = self.model_path / "onnx" / f"{model_prefix}_image_encoder.onnx"
        
        if text_onnx_path.exists() and image_onnx_path.exists():
            return True
        
        # Fallback to old naming convention
        onnx_path = self.model_path / "onnx" / "visual_encoder.onnx"
        text_onnx_path = self.model_path / "onnx" / "text_encoder.onnx"
        return onnx_path.exists() and text_onnx_path.exists()
    
    async def _load_onnx_model(self):
        """Load ONNX optimized model (supports new automated format)"""
        logger.info("Loading ONNX model...")
        
        # ONNX Runtime providers
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        # Try new automated format first
        model_prefix = f"clip_{self.model_name.replace('/', '_')}"
        text_model_path = self.model_path / "onnx" / f"{model_prefix}_text_encoder.onnx"
        image_model_path = self.model_path / "onnx" / f"{model_prefix}_image_encoder.onnx"
        
        if text_model_path.exists() and image_model_path.exists():
            logger.info("Using automated ONNX models...")
            self.text_session = ort.InferenceSession(str(text_model_path), providers=providers)
            self.visual_session = ort.InferenceSession(str(image_model_path), providers=providers)
        else:
            # Fallback to old naming convention
            logger.info("Using legacy ONNX models...")
            visual_model_path = self.model_path / "onnx" / "visual_encoder.onnx"
            text_model_path = self.model_path / "onnx" / "text_encoder.onnx"
            self.visual_session = ort.InferenceSession(str(visual_model_path), providers=providers)
            self.text_session = ort.InferenceSession(str(text_model_path), providers=providers)
        
        # Load tokenizer
        tokenizer_path = self.model_path / "tokenizer.json"
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r') as f:
                self.tokenizer_config = json.load(f)
        else:
            # Fallback to CLIP tokenizer
            _, self.clip_preprocess = clip.load(self.model_name, device="cpu")
            self.tokenizer = clip.tokenize
        
        self.use_onnx = True
        logger.info("ONNX model loaded successfully")
    
    async def _load_pytorch_model(self):
        """Load PyTorch CLIP model"""
        logger.info("Loading PyTorch CLIP model...")
        
        loop = asyncio.get_event_loop()
        self.clip_model, self.clip_preprocess = await loop.run_in_executor(
            self.executor,
            clip.load,
            self.model_name,
            self.device
        )
        
        self.clip_model.eval()
        self.embedding_dim = self.clip_model.visual.output_dim
        
        logger.info("PyTorch model loaded successfully")
    
    async def generate_embeddings(self, keyframes: List[Keyframe]) -> List[Embedding]:
        """
        Generate embeddings for keyframes
        
        Args:
            keyframes: List of keyframes to process
            
        Returns:
            List of embeddings
        """
        logger.info(f"Generating embeddings for {len(keyframes)} keyframes")
        start_time = time.time()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(keyframes), self.batch_size):
            batch = keyframes[i:i + self.batch_size]
            
            try:
                batch_embeddings = await self._process_batch(batch)
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(keyframes)-1)//self.batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                # Create zero embeddings for failed batch
                for keyframe in batch:
                    zero_embedding = Embedding(
                        keyframe_id=keyframe.keyframe_id,
                        embedding=np.zeros(self.embedding_dim),
                        dimension=self.embedding_dim,
                        model_name=self.model_name,
                        creation_time=time.time()
                    )
                    embeddings.append(zero_embedding)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f}s")
        
        return embeddings
    
    async def _process_batch(self, keyframes: List[Keyframe]) -> List[Embedding]:
        """Process a batch of keyframes"""
        if self.use_onnx:
            return await self._process_batch_onnx(keyframes)
        else:
            return await self._process_batch_pytorch(keyframes)
    
    async def _process_batch_pytorch(self, keyframes: List[Keyframe]) -> List[Embedding]:
        """Process batch using PyTorch model"""
        loop = asyncio.get_event_loop()
        
        # Prepare images
        images = []
        for keyframe in keyframes:
            if keyframe.image_data is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(keyframe.image_data, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            else:
                # Load from file
                image_pil = Image.open(keyframe.image_path).convert('RGB')
            
            # Preprocess
            image_tensor = self.clip_preprocess(image_pil)
            images.append(image_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(images).to(self.device)
        
        # Generate embeddings
        embeddings_data = await loop.run_in_executor(
            self.executor,
            self._encode_images_pytorch,
            batch_tensor
        )
        
        # Create embedding objects
        embeddings = []
        for i, keyframe in enumerate(keyframes):
            embedding = Embedding(
                keyframe_id=keyframe.keyframe_id,
                embedding=embeddings_data[i],
                dimension=self.embedding_dim,
                model_name=self.model_name,
                creation_time=time.time()
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def _encode_images_pytorch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Encode images using PyTorch model"""
        with torch.no_grad():
            features = self.clip_model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            return features.cpu().numpy()
    
    async def _process_batch_onnx(self, keyframes: List[Keyframe]) -> List[Embedding]:
        """Process batch using ONNX model"""
        loop = asyncio.get_event_loop()
        
        # Prepare images for ONNX
        images = []
        for keyframe in keyframes:
            if keyframe.image_data is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(keyframe.image_data, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            else:
                # Load from file
                image_pil = Image.open(keyframe.image_path).convert('RGB')
            
            # Preprocess for ONNX (similar to CLIP preprocessing)
            image_array = self._preprocess_for_onnx(image_pil)
            images.append(image_array)
        
        # Stack into batch
        batch_array = np.stack(images)
        
        # Run ONNX inference
        embeddings_data = await loop.run_in_executor(
            self.executor,
            self._encode_images_onnx,
            batch_array
        )
        
        # Create embedding objects
        embeddings = []
        for i, keyframe in enumerate(keyframes):
            embedding = Embedding(
                keyframe_id=keyframe.keyframe_id,
                embedding=embeddings_data[i],
                dimension=self.embedding_dim,
                model_name=self.model_name,
                creation_time=time.time()
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def _preprocess_for_onnx(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize and center crop
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with CLIP stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        image_array = (image_array - mean) / std
        
        # Convert to CHW format
        image_array = image_array.transpose(2, 0, 1)
        
        return image_array
    
    def _encode_images_onnx(self, batch_array: np.ndarray) -> np.ndarray:
        """Encode images using ONNX model"""
        input_name = self.visual_session.get_inputs()[0].name
        output_name = self.visual_session.get_outputs()[0].name
        
        # Run inference
        result = self.visual_session.run([output_name], {input_name: batch_array})
        features = result[0]
        
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / norms
        
        return features
    
    async def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to embedding
        
        Args:
            text: Text query to encode
            
        Returns:
            Text embedding vector
        """
        if self.use_onnx:
            return await self._encode_text_onnx(text)
        else:
            return await self._encode_text_pytorch(text)
    
    async def _encode_text_pytorch(self, text: str) -> np.ndarray:
        """Encode text using PyTorch model"""
        loop = asyncio.get_event_loop()
        
        # Tokenize text
        text_tokens = clip.tokenize([text]).to(self.device)
        
        # Encode
        embedding = await loop.run_in_executor(
            self.executor,
            self._encode_text_pytorch_sync,
            text_tokens
        )
        
        return embedding
    
    def _encode_text_pytorch_sync(self, text_tokens: torch.Tensor) -> np.ndarray:
        """Synchronous text encoding with PyTorch"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
    
    async def _encode_text_onnx(self, text: str) -> np.ndarray:
        """Encode text using ONNX model"""
        loop = asyncio.get_event_loop()
        
        # Tokenize (simplified - would need proper tokenizer for production)
        tokens = self._tokenize_for_onnx(text)
        
        # Encode
        embedding = await loop.run_in_executor(
            self.executor,
            self._encode_text_onnx_sync,
            tokens
        )
        
        return embedding
    
    def _tokenize_for_onnx(self, text: str) -> np.ndarray:
        """Tokenize text for ONNX model"""
        # Simplified tokenization - would need proper implementation
        # This is a placeholder that uses CLIP tokenization
        if hasattr(self, 'tokenizer'):
            tokens = clip.tokenize([text])
            return tokens.numpy()
        else:
            # Fallback to simple encoding
            return np.array([[1] + [ord(c) % 49407 for c in text[:76]] + [2] + [0] * (77 - len(text) - 2)])
    
    def _encode_text_onnx_sync(self, tokens: np.ndarray) -> np.ndarray:
        """Synchronous text encoding with ONNX"""
        input_name = self.text_session.get_inputs()[0].name
        output_name = self.text_session.get_outputs()[0].name
        
        # Run inference
        result = self.text_session.run([output_name], {input_name: tokens})
        features = result[0]
        
        # Normalize
        norm = np.linalg.norm(features)
        features = features / norm
        
        return features[0]
    
    async def compute_similarity(self, 
                               text_embedding: np.ndarray, 
                               image_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarities between text and image embeddings
        
        Args:
            text_embedding: Text query embedding
            image_embeddings: List of image embeddings
            
        Returns:
            Array of similarity scores
        """
        # Stack image embeddings
        image_matrix = np.stack(image_embeddings)
        
        # Compute cosine similarities
        similarities = np.dot(image_matrix, text_embedding)
        
        return similarities
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "use_onnx": self.use_onnx,
            "batch_size": self.batch_size
        }
