from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SceneAnalysis:
    scene_type: str = "Unknown"  # Portrait, Landscape, Architecture, Indoor, Unknown
    primary_subject_detected: bool = False
    primary_subject_box: Optional[Tuple[int, int, int, int]] = None  # (ymin, xmin, ymax, xmax)
    depth_confidence: float = 0.5  # 0.0 to 1.0
    disocclusion_risk: str = "Low"  # Low, Medium, High
    subject_deformation_risk: str = "Low"  # Low, Medium, High
    quality_indicator: str = "Good"  # Excellent, Good, Limited
    safety_status: str = "SAFE"  # SAFE, REDUCED, UNSUPPORTED
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MotionIntent:
    movement_style: str = "Cinematic Auto"  # Cinematic Auto, Dolly Push, Zoom-in, Orbit, Micro-orbit
    strength_level: str = "Cinematic"  # Subtle, Cinematic, Dynamic, Dramatic
    motion_direction: str = "Auto"  # Auto, Pan Left, Pan Right, Orbit CW, Orbit CCW
    duration: float = 4.0  # Duration in seconds
    loop: bool = True

@dataclass
class MotionPlan:
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # list of camera states (x, y, z, pitch, yaw, t)
    target_screen_disparity: float = 0.0
    predicted_reconstruction_ratio: float = 0.0
    actual_strength_multiplier: float = 1.0
    was_reduced: bool = False

@dataclass
class PreviewConfig:
    resolution: Tuple[int, int] = (640, 360)
    num_frames: int = 36
    quality_level: str = "Balanced"  # Fast, Balanced, Quality

@dataclass
class PreviewCacheKey:
    image_hash: str
    depth_hash: str
    scene_analysis_hash: str
    movement_style: str
    strength_level: str
    motion_direction: str
    duration: float
    loop: bool
    quality_level: str

@dataclass
class PreviewResult:
    video_url: str
    frame_urls: List[str]
    fps: float
    duration: float
    quality: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
