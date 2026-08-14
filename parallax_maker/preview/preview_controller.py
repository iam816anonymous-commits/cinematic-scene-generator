from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from .preview_models import SceneAnalysis, MotionIntent, MotionPlan, PreviewConfig, PreviewResult
from .preview_cache import PreviewCache
from .preview_renderer import PreviewRenderer
from ..controller import AppState

class PreviewController:
    _scene_analysis_cache: Dict[str, SceneAnalysis] = {}
    _motion_plan_cache: Dict[str, MotionPlan] = {}

    @classmethod
    def get_scene_analysis(cls, state: AppState) -> SceneAnalysis:
        """Retrieves or computes the automatic Scene Analysis for the current AppState."""
        if state is None or state.imgData is None or state.depthMapData is None:
            return SceneAnalysis()

        # Compute image & depth hashes to detect if either has changed
        img_hash = PreviewCache.compute_image_hash(state.imgData)
        depth_hash = PreviewCache.compute_image_hash(state.depthMapData)
        combined_hash = f"{img_hash}_{depth_hash}"

        if combined_hash in cls._scene_analysis_cache:
            return cls._scene_analysis_cache[combined_hash]

        # Convert PIL image to numpy RGB
        image_np = np.array(state.imgData.convert("RGB"))
        depth_np = state.depthMapData

        # Compute Scene Analysis on the fly
        analysis = PreviewRenderer.analyze_scene(image_np, depth_np)
        cls._scene_analysis_cache[combined_hash] = analysis
        return analysis

    @classmethod
    def get_motion_plan(cls, state: AppState, intent: MotionIntent) -> MotionPlan:
        """Retrieves or computes the automatic Motion Plan for the current AppState and MotionIntent."""
        analysis = cls.get_scene_analysis(state)

        intent_key = f"{state.filename}_{intent.movement_style}_{intent.strength_level}_{intent.motion_direction}_{intent.duration}_{intent.loop}"
        if intent_key in cls._motion_plan_cache:
            return cls._motion_plan_cache[intent_key]

        plan = PreviewRenderer.plan_motion(analysis, intent)
        cls._motion_plan_cache[intent_key] = plan
        return plan

    @classmethod
    def generate_preview(cls, state: AppState, intent: MotionIntent, quality: str = "Balanced") -> PreviewResult:
        """
        Coordinates caches, performs safety checks, and triggers the adaptive low-res preview rendering.
        Ensures the preview result is fully loopable and is generated in under a few seconds!
        """
        if state is None or state.imgData is None or state.depthMapData is None:
            raise ValueError("No image or depth map available. Please upload an image and generate a depth map first.")

        # 1. Hashes & Cache Lookup
        img_hash = PreviewCache.compute_image_hash(state.imgData)
        depth_hash = PreviewCache.compute_image_hash(state.depthMapData)

        analysis = cls.get_scene_analysis(state)
        scene_analysis_hash = PreviewCache.compute_image_hash(str(analysis.diagnostics))

        cache_key = PreviewCache.get_cache_key(
            image_hash=img_hash,
            depth_hash=depth_hash,
            scene_analysis_hash=scene_analysis_hash,
            movement_style=intent.movement_style,
            strength_level=intent.strength_level,
            motion_direction=intent.motion_direction,
            duration=intent.duration,
            loop=intent.loop,
            quality_level=quality
        )

        cached_result = PreviewCache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 2. Get the Motion Plan (which handles safety checking internally)
        motion_plan = cls.get_motion_plan(state, intent)

        # 3. Choose Preview Config
        num_frames = 24 if quality == "Fast" else (36 if quality == "Balanced" else 48)
        preview_config = PreviewConfig(
            resolution=(640, 360),
            num_frames=num_frames,
            quality_level=quality
        )

        # 4. Trigger the low-resolution render
        # Slices, masks, and camera parameters are converted to target space natively
        image_np = np.array(state.imgData.convert("RGB"))
        depth_np = state.depthMapData
        state_dir = Path(state.filename)

        result = PreviewRenderer.render_preview(
            image=image_np,
            depth_map=depth_np,
            scene_analysis=analysis,
            motion_plan=motion_plan,
            preview_config=preview_config,
            output_dir=state_dir
        )

        # Cache result
        PreviewCache.set(cache_key, result)
        return result
