"""
Intelligent Query Enhancer with Multi-Modal Feature Understanding
Combines LLM-based prompt engineering with feature-aware query enhancement
"""

import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
import re

# Lightweight LLM options
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Analysis of user query intent and components."""
    original_query: str
    enhanced_query: str
    query_type: str  # action, object, scene, person, complex
    entities: List[str]
    actions: List[str]
    scene_context: List[str]
    temporal_context: List[str]
    confidence: float

@dataclass
class EnhancementRules:
    """Domain-specific enhancement rules for video content."""
    object_expansions: Dict[str, List[str]]
    action_contexts: Dict[str, List[str]]
    scene_descriptors: Dict[str, List[str]]
    temporal_markers: Dict[str, List[str]]

class LightweightQueryEnhancer:
    """
    High-performance query enhancer using local lightweight LLM.
    Focuses on video-specific query enhancement with caching.
    """
    
    def __init__(self, 
                 settings = None,
                 model_name: str = "microsoft/DialoGPT-small",
                 cache_ttl: int = 3600,
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        """
        Initialize the query enhancer.
        
        Args:
            settings: Settings object (if provided, overrides individual params)
            model_name: HuggingFace model for text generation
            cache_ttl: Cache time-to-live in seconds
            redis_host: Redis host for caching
            redis_port: Redis port
        """
        # Use settings if provided
        if settings:
            redis_host = getattr(settings, 'REDIS_HOST', redis_host)
            redis_port = getattr(settings, 'REDIS_PORT', redis_port)
            
        self.model_name = model_name
        self.cache_ttl = cache_ttl
        self.device = torch.device("cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.text_pipeline = None
        
        # Initialize cache
        try:
            self.cache = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.cache.ping()
            logger.info("Connected to Redis cache for query enhancement")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.cache = None
        
        # Load enhancement rules
        self.rules = self._load_enhancement_rules()
        
        # Initialize models - do this synchronously to avoid async in __init__
        self._initialize_models_sync()
    
    def _initialize_models_sync(self):
        """Initialize models synchronously to avoid async in __init__"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using template-based enhancement only")
            return
        
        try:
            logger.info(f"Loading lightweight model: {self.model_name}")
            
            # Use a lightweight text generation pipeline
            self.text_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device.type == "cuda" else -1,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=50256
            )
            
            logger.info("Query enhancement model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load enhancement model: {e}")
            logger.info("Falling back to template-based enhancement")
    
    async def _initialize_models(self):
        """Async initialization method for backwards compatibility"""
        self._initialize_models_sync()
    
    def _load_enhancement_rules(self) -> EnhancementRules:
        """Load video-specific enhancement rules."""
        return EnhancementRules(
            object_expansions={
                "dog": ["dog", "canine", "pet", "animal", "furry", "tail wagging", "running", "playing"],
                "car": ["car", "vehicle", "automobile", "driving", "road", "traffic", "wheels", "engine"],
                "person": ["person", "human", "people", "individual", "man", "woman", "walking", "standing"],
                "food": ["food", "eating", "meal", "cooking", "kitchen", "ingredients", "preparation"],
                "meeting": ["meeting", "conference", "business", "discussion", "presentation", "office", "table"],
                "sports": ["sports", "athletic", "game", "competition", "players", "field", "stadium"],
                "nature": ["nature", "outdoor", "trees", "landscape", "natural", "environment", "wildlife"],
                "music": ["music", "concert", "performance", "instruments", "singing", "stage", "audience"]
            },
            action_contexts={
                "running": ["running", "jogging", "moving fast", "exercise", "outdoor activity", "motion"],
                "cooking": ["cooking", "preparing food", "kitchen activity", "chef", "ingredients", "stove"],
                "meeting": ["meeting", "discussing", "business conversation", "presentation", "collaboration"],
                "playing": ["playing", "recreational activity", "fun", "games", "entertainment", "leisure"],
                "driving": ["driving", "operating vehicle", "transportation", "road travel", "journey"],
                "speaking": ["speaking", "talking", "conversation", "communication", "presentation", "dialogue"]
            },
            scene_descriptors={
                "indoor": ["indoor", "inside", "interior", "room", "building", "enclosed space"],
                "outdoor": ["outdoor", "outside", "exterior", "open air", "natural setting", "landscape"],
                "office": ["office", "workplace", "business environment", "desk", "computer", "professional"],
                "kitchen": ["kitchen", "cooking area", "food preparation", "appliances", "counter", "dining"],
                "park": ["park", "public space", "recreation area", "trees", "grass", "outdoor leisure"],
                "street": ["street", "road", "urban setting", "city", "traffic", "buildings", "sidewalk"]
            },
            temporal_markers={
                "morning": ["morning", "early", "sunrise", "dawn", "beginning of day"],
                "evening": ["evening", "sunset", "dusk", "end of day", "twilight"],
                "night": ["night", "dark", "nighttime", "after dark", "evening hours"],
                "day": ["day", "daylight", "daytime", "bright", "sunny", "during the day"]
            }
        )
    
    async def enhance_query(self, 
                          query: str, 
                          context: Optional[Dict] = None,
                          use_llm: bool = True) -> QueryAnalysis:
        """
        Enhance a user query for better video search results.
        
        Args:
            query: Original user query
            context: Optional context (user preferences, previous queries, etc.)
            use_llm: Whether to use LLM enhancement or template-based only
            
        Returns:
            QueryAnalysis with enhanced query and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(query, context)
        if self.cache:
            cached = await self._get_cached_result(cache_key)
            if cached:
                return cached
        
        # Analyze query structure
        analysis = await self._analyze_query_structure(query)
        
        # Template-based enhancement (always applied)
        template_enhanced = await self._template_enhance(query, analysis)
        
        # LLM enhancement (if available and requested)
        if use_llm and self.text_pipeline:
            llm_enhanced = await self._llm_enhance(template_enhanced, analysis)
        else:
            llm_enhanced = template_enhanced
        
        # Create final analysis
        result = QueryAnalysis(
            original_query=query,
            enhanced_query=llm_enhanced,
            query_type=analysis["type"],
            entities=analysis["entities"],
            actions=analysis["actions"],
            scene_context=analysis["scene_context"],
            temporal_context=analysis["temporal_context"],
            confidence=analysis["confidence"]
        )
        
        # Cache result
        if self.cache:
            await self._cache_result(cache_key, result)
        
        return result
    
    async def _analyze_query_structure(self, query: str) -> Dict[str, Any]:
        """Analyze the structure and intent of the query."""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Classify query type
        query_type = "complex"
        if any(word in self.rules.object_expansions for word in words):
            query_type = "object"
        elif any(word in self.rules.action_contexts for word in words):
            query_type = "action"
        elif any(word in self.rules.scene_descriptors for word in words):
            query_type = "scene"
        elif any(word in ["person", "people", "man", "woman", "human"] for word in words):
            query_type = "person"
        
        # Extract entities
        entities = []
        for word in words:
            if word in self.rules.object_expansions:
                entities.extend(self.rules.object_expansions[word][:3])  # Top 3
        
        # Extract actions
        actions = []
        for word in words:
            if word in self.rules.action_contexts:
                actions.extend(self.rules.action_contexts[word][:3])
        
        # Extract scene context
        scene_context = []
        for scene, descriptors in self.rules.scene_descriptors.items():
            if any(desc in query_lower for desc in descriptors):
                scene_context.extend(descriptors[:2])
        
        # Extract temporal context
        temporal_context = []
        for time, markers in self.rules.temporal_markers.items():
            if any(marker in query_lower for marker in markers):
                temporal_context.extend(markers[:2])
        
        return {
            "type": query_type,
            "entities": list(set(entities)),
            "actions": list(set(actions)),
            "scene_context": list(set(scene_context)),
            "temporal_context": list(set(temporal_context)),
            "confidence": 0.8 if entities or actions else 0.6
        }
    
    async def _template_enhance(self, query: str, analysis: Dict) -> str:
        """Apply template-based enhancement rules."""
        enhanced_parts = [query]
        
        # Add entity expansions
        if analysis["entities"]:
            enhanced_parts.extend(analysis["entities"][:5])
        
        # Add action contexts
        if analysis["actions"]:
            enhanced_parts.extend(analysis["actions"][:3])
        
        # Add scene descriptors
        if analysis["scene_context"]:
            enhanced_parts.extend(analysis["scene_context"][:3])
        
        # Add temporal context
        if analysis["temporal_context"]:
            enhanced_parts.extend(analysis["temporal_context"][:2])
        
        return ", ".join(enhanced_parts)
    
    async def _llm_enhance(self, template_enhanced: str, analysis: Dict) -> str:
        """Use LLM to further enhance the query."""
        if not self.text_pipeline:
            return template_enhanced
        
        try:
            # Create prompt for video-specific enhancement
            prompt = self._create_enhancement_prompt(template_enhanced, analysis)
            
            # Generate enhancement
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._generate_enhancement, 
                prompt
            )
            
            return self._parse_llm_output(result, template_enhanced)
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return template_enhanced
    
    def _create_enhancement_prompt(self, query: str, analysis: Dict) -> str:
        """Create a prompt for LLM enhancement."""
        prompt = f"""Enhance this video search query by adding relevant visual details and context:

Original query: {query}
Query type: {analysis['type']}

Enhanced query with visual details:"""
        
        return prompt
    
    def _generate_enhancement(self, prompt: str) -> str:
        """Generate enhancement using the LLM pipeline."""
        try:
            result = self.text_pipeline(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return prompt
    
    def _parse_llm_output(self, llm_output: str, fallback: str) -> str:
        """Parse and clean LLM output."""
        try:
            # Extract the enhanced part after the prompt
            lines = llm_output.split('\n')
            enhanced_line = None
            
            for line in lines:
                if line.strip() and not line.startswith('Original query:') and not line.startswith('Query type:'):
                    enhanced_line = line.strip()
                    break
            
            if enhanced_line and len(enhanced_line) > 10:
                return enhanced_line
            else:
                return fallback
                
        except Exception:
            return fallback
    
    def _get_cache_key(self, query: str, context: Optional[Dict]) -> str:
        """Generate cache key for query and context."""
        context_str = json.dumps(context or {}, sort_keys=True)
        return hashlib.md5(f"{query}:{context_str}".encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[QueryAnalysis]:
        """Get cached enhancement result."""
        try:
            cached = self.cache.get(f"query_enhance:{cache_key}")
            if cached:
                data = json.loads(cached)
                return QueryAnalysis(**data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: QueryAnalysis):
        """Cache enhancement result."""
        try:
            data = {
                "original_query": result.original_query,
                "enhanced_query": result.enhanced_query,
                "query_type": result.query_type,
                "entities": result.entities,
                "actions": result.actions,
                "scene_context": result.scene_context,
                "temporal_context": result.temporal_context,
                "confidence": result.confidence
            }
            self.cache.setex(
                f"query_enhance:{cache_key}",
                self.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    async def learn_from_feedback(self, 
                                query: str, 
                                enhanced_query: str,
                                user_feedback: Dict):
        """Learn from user feedback to improve enhancement rules."""
        # This could be implemented to update enhancement rules
        # based on user click-through rates, relevance feedback, etc.
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return {
            "model_loaded": self.text_pipeline is not None,
            "cache_available": self.cache is not None,
            "rules_count": {
                "objects": len(self.rules.object_expansions),
                "actions": len(self.rules.action_contexts),
                "scenes": len(self.rules.scene_descriptors),
                "temporal": len(self.rules.temporal_markers)
            }
        }
