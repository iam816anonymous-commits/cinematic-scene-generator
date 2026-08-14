import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .preview_models import PreviewCacheKey, PreviewResult

class PreviewCache:
    _in_memory_cache: Dict[str, PreviewResult] = {}

    @staticmethod
    def compute_image_hash(image_data) -> str:
        """Computes a SHA256 hash of an image (PIL Image or numpy array)."""
        if image_data is None:
            return "empty_image"
        try:
            import numpy as np
            if hasattr(image_data, "tobytes"):
                # numpy array
                data_bytes = image_data.tobytes()
            elif hasattr(image_data, "convert"):
                # PIL Image
                data_bytes = image_data.tobytes()
            else:
                data_bytes = str(image_data).encode("utf-8")
            return hashlib.sha256(data_bytes).hexdigest()
        except Exception:
            return "fallback_image_hash"

    @staticmethod
    def get_cache_key(
        image_hash: str,
        depth_hash: str,
        scene_analysis_hash: str,
        movement_style: str,
        strength_level: str,
        motion_direction: str,
        duration: float,
        loop: bool,
        quality_level: str
    ) -> str:
        """Generates a unique deterministic string key from the preview parameters."""
        raw_key = f"{image_hash}_{depth_hash}_{scene_analysis_hash}_{movement_style}_{strength_level}_{motion_direction}_{duration}_{loop}_{quality_level}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, key: str) -> Optional[PreviewResult]:
        """Retrieves a cached preview result if it exists and the physical files are intact."""
        res = cls._in_memory_cache.get(key)
        if res is not None:
            # Verify that the files pointed to by the URLs actually exist on disk
            from urllib.parse import urlparse
            import os
            # Convert URL to local path
            parsed = urlparse(res.video_url)
            # Remove /tmp-images/ prefix to get local path relative to root
            local_path_str = parsed.path.replace("/tmp-images/", "")
            local_path = Path(local_path_str)
            if local_path.exists():
                res.cache_hit = True
                return res
        return None

    @classmethod
    def set(cls, key: str, result: PreviewResult) -> None:
        """Stores a preview result in the cache."""
        cls._in_memory_cache[key] = result

    @classmethod
    def invalidate_all(cls) -> None:
        """Clears all cached previews."""
        cls._in_memory_cache.clear()
