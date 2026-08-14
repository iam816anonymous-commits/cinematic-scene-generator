# (c) 2024 Niels Provos

class SceneAnalysis:
    def __init__(
        self,
        scene_type="Square/Generic",
        primary_subject_depth=128.0,
        depth_confidence=0.8,
        disocclusion_risk=0.2,
        reconstruction_risk=0.2,
        motion_safety="Excellent"
    ):
        self.scene_type = scene_type
        self.primary_subject_depth = primary_subject_depth
        self.depth_confidence = depth_confidence
        self.disocclusion_risk = disocclusion_risk
        self.reconstruction_risk = reconstruction_risk
        self.motion_safety = motion_safety

    def to_dict(self):
        return {
            "scene_type": self.scene_type,
            "primary_subject_depth": self.primary_subject_depth,
            "depth_confidence": self.depth_confidence,
            "disocclusion_risk": self.disocclusion_risk,
            "reconstruction_risk": self.reconstruction_risk,
            "motion_safety": self.motion_safety
        }


class MotionIntent:
    def __init__(
        self,
        style="Cinematic Auto",
        strength="Cinematic",
        duration=4,
        loop=True
    ):
        self.style = style
        self.strength = strength
        self.duration = duration
        self.loop = loop

    def to_dict(self):
        return {
            "style": self.style,
            "strength": self.strength,
            "duration": self.duration,
            "loop": self.loop
        }


class MotionPlan:
    def __init__(
        self,
        parallax_strength=0.55,
        camera_motion_strength=0.3,
        zoom_strength=0.2,
        rotation_strength=0.15,
        push_distance=100.0,
        num_frames=36,
        width=640,
        height=360,
        motion_reduced=False
    ):
        self.parallax_strength = parallax_strength
        self.camera_motion_strength = camera_motion_strength
        self.zoom_strength = zoom_strength
        self.rotation_strength = rotation_strength
        self.push_distance = push_distance
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.motion_reduced = motion_reduced

    def to_dict(self):
        return {
            "parallax_strength": self.parallax_strength,
            "camera_motion_strength": self.camera_motion_strength,
            "zoom_strength": self.zoom_strength,
            "rotation_strength": self.rotation_strength,
            "push_distance": self.push_distance,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "motion_reduced": self.motion_reduced
        }


class PreviewConfig:
    def __init__(self, quality="Balanced"):
        self.quality = quality


class PreviewResult:
    def __init__(
        self,
        frames=None,
        fps=24,
        duration=4.0,
        quality="Balanced",
        motion_plan_id="",
        diagnostics=None,
        cache_hit=False
    ):
        self.frames = frames if frames is not None else []
        self.fps = fps
        self.duration = duration
        self.quality = quality
        self.motion_plan_id = motion_plan_id
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.cache_hit = cache_hit


class PreviewCacheKey:
    def __init__(
        self,
        image_hash,
        depth_hash,
        scene_analysis_hash,
        motion_style,
        strength,
        duration,
        loop,
        quality
    ):
        self.image_hash = image_hash
        self.depth_hash = depth_hash
        self.scene_analysis_hash = scene_analysis_hash
        self.motion_style = motion_style
        self.strength = strength
        self.duration = duration
        self.loop = loop
        self.quality = quality

    def to_tuple(self):
        return (
            self.image_hash,
            self.depth_hash,
            self.scene_analysis_hash,
            self.motion_style,
            self.strength,
            self.duration,
            self.loop,
            self.quality,
        )

    def __eq__(self, other):
        if not isinstance(other, PreviewCacheKey):
            return False
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())
