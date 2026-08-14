# (c) 2024 Niels Provos

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from ..camera import Camera
from ..slice import ImageSlice
from ..segmentation import render_view
from .preview_models import SceneAnalysis, MotionPlan, PreviewResult

def analyze_scene(image_rgb, depth_map) -> SceneAnalysis:
    """
    Automated Scene Analysis.
    Analyzes image dimensions, depth map statistics, and returns a SceneAnalysis.
    """
    if image_rgb is None or depth_map is None:
        return SceneAnalysis()

    h, w = image_rgb.shape[:2]
    aspect_ratio = w / h

    # Determine scene type
    if aspect_ratio > 1.35:
        scene_type = "Landscape"
    elif aspect_ratio < 0.75:
        scene_type = "Portrait"
    else:
        scene_type = "Square/Generic"

    # Analyze depth stats
    depth_std = float(np.std(depth_map))
    depth_mean = float(np.mean(depth_map))
    depth_max = float(np.max(depth_map))
    depth_min = float(np.min(depth_map))

    depth_confidence = min(1.0, max(0.1, depth_std / 64.0))

    # Calculate disocclusion risk: high if high depth range next to dark background
    disocclusion_risk = min(1.0, max(0.0, (depth_max - depth_min) / 255.0 * (1.0 - depth_mean / 255.0)))
    reconstruction_risk = disocclusion_risk

    # Determine motion safety
    if disocclusion_risk < 0.35 and depth_confidence > 0.5:
        motion_safety = "Excellent"
    elif disocclusion_risk < 0.65 and depth_confidence > 0.3:
        motion_safety = "Good"
    else:
        motion_safety = "Limited"

    # Assume primary subject is around the upper-middle depth band
    primary_subject_depth = float(np.percentile(depth_map, 75))

    return SceneAnalysis(
        scene_type=scene_type,
        primary_subject_depth=primary_subject_depth,
        depth_confidence=depth_confidence,
        disocclusion_risk=disocclusion_risk,
        reconstruction_risk=reconstruction_risk,
        motion_safety=motion_safety
    )


def plan_motion(scene_analysis: SceneAnalysis, motion_intent, quality="Balanced") -> MotionPlan:
    """
    Automatic Motion Planning.
    Translates high-level SceneAnalysis and MotionIntent into lower-level MotionPlan parameters.
    """
    strength_name = motion_intent.strength
    style_name = motion_intent.style

    # Base low-level multipliers based on high-level strength
    if strength_name == "Subtle":
        base_strength = 0.25
        zoom_mult = 0.1
        rot_mult = 0.05
    elif strength_name == "Dynamic":
        base_strength = 0.75
        zoom_mult = 0.5
        rot_mult = 0.3
    elif strength_name == "Dramatic":
        base_strength = 1.00
        zoom_mult = 0.75
        rot_mult = 0.45
    else: # Cinematic (default)
        base_strength = 0.55
        zoom_mult = 0.25
        rot_mult = 0.15

    # Style adaptations
    parallax_strength = base_strength
    camera_motion_strength = base_strength
    zoom_strength = base_strength * zoom_mult
    rotation_strength = base_strength * rot_mult

    # Adjust based on style name
    if style_name == "Horizontal Pan":
        camera_motion_strength *= 1.5
        zoom_strength = 0.0
        rotation_strength = 0.0
    elif style_name == "Vertical Pan":
        camera_motion_strength *= 1.5
        zoom_strength = 0.0
        rotation_strength = 0.0
    elif style_name == "Push-In (Zoom)":
        camera_motion_strength = 0.0
        zoom_strength *= 2.0
        rotation_strength = 0.0
    elif style_name == "Orbit Dolly":
        camera_motion_strength *= 0.5
        zoom_strength *= 0.5
        rotation_strength *= 2.0

    # Safety Validation: clamp/reduce if disocclusion_risk is too high
    motion_reduced = False
    if scene_analysis.disocclusion_risk > 0.6 and strength_name in ["Dynamic", "Dramatic"]:
        parallax_strength *= 0.6
        camera_motion_strength *= 0.6
        zoom_strength *= 0.6
        rotation_strength *= 0.6
        motion_reduced = True

    # Frame count based on quality
    if quality == "Fast":
        num_frames = 24
    elif quality == "Quality":
        num_frames = 48
    else: # Balanced
        num_frames = 36

    # Aspect-ratio preserving target dimensions
    # Default target height: Fast=270, Balanced=360, Quality=480
    if quality == "Fast":
        target_h = 180
    elif quality == "Quality":
        target_h = 480
    else: # Balanced
        target_h = 270

    # Compute width based on aspect ratio (e.g., from Landscape, Portrait)
    if scene_analysis.scene_type == "Landscape":
        width = int(target_h * 16.0 / 9.0)
    elif scene_analysis.scene_type == "Portrait":
        width = int(target_h * 9.0 / 16.0)
    else:
        width = target_h # square-ish

    # Ensure even dimensions for OpenCV remap/resize
    width = (width // 2) * 2
    height = (target_h // 2) * 2

    # Push distance based on camera distance (typically matches production distance)
    push_distance = 100.0 * 0.75

    return MotionPlan(
        parallax_strength=parallax_strength,
        camera_motion_strength=camera_motion_strength,
        zoom_strength=zoom_strength,
        rotation_strength=rotation_strength,
        push_distance=push_distance,
        num_frames=num_frames,
        width=width,
        height=height,
        motion_reduced=motion_reduced
    )


def render_preview_sequence(state, motion_plan: MotionPlan, motion_intent, progress_callback=None) -> PreviewResult:
    """
    Renders the preview sequence using exactly the same rendering pipeline at lower resolution/frames.
    """
    if not state.image_slices or len(state.image_slices) == 0:
        raise ValueError("No image slices available for preview rendering.")

    width, height = motion_plan.width, motion_plan.height
    num_frames = motion_plan.num_frames
    push_distance = motion_plan.push_distance

    # Create temporary low-resolution copies of slices to make rendering blazing fast
    resized_reconstructed_slices = []
    resized_original_slices = []

    # Get reconstructed slices (usually padded)
    margin = 0.1
    reconstructed_slices = state.get_reconstructed_slices(margin=margin, use_ai=False)

    for slice_img in reconstructed_slices:
        h, w = slice_img.image.shape[:2]
        # Calculate matching low-resolution dimensions preserving padding relative scale
        scale_x = width / (state.imgData.size[0])
        scale_y = height / (state.imgData.size[1])
        low_w = int(w * scale_x)
        low_h = int(h * scale_y)
        low_w = (low_w // 2) * 2
        low_h = (low_h // 2) * 2

        low_res_img = cv2.resize(slice_img.image, (low_w, low_h), interpolation=cv2.INTER_AREA)
        resized_reconstructed_slices.append(ImageSlice(image=low_res_img, depth=slice_img.depth))

    # Resize unpadded original slices
    for slice_img in state.image_slices:
        h, w = slice_img.image.shape[:2]
        scale_x = width / w
        scale_y = height / h
        low_w = int(w * scale_x)
        low_h = int(h * scale_y)
        low_w = (low_w // 2) * 2
        low_h = (low_h // 2) * 2

        low_res_img = cv2.resize(slice_img.image, (low_w, low_h), interpolation=cv2.INTER_AREA)
        resized_original_slices.append(ImageSlice(image=low_res_img, depth=slice_img.depth))

    # Camera model
    camera = Camera(state.camera.camera_distance, state.camera.focal_length, state.camera.max_distance)
    camera_matrix = camera.camera_matrix(width, height)

    # Scale 3D card geometry matching the resized padded dimensions
    card_corners_3d_list = []
    for image_slice in resized_reconstructed_slices:
        card = image_slice.create_card(height, width, camera)
        card[:, :2] *= (1.0 + 2.0 * margin)
        card_corners_3d_list.append(card)

    start_camera_position = np.array([0, 0, -camera.camera_distance], dtype=np.float32)

    output_dir = Path(state.filename)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    preview_frame_paths = []
    for i in range(num_frames):
        requested_pos = start_camera_position.copy()

        # Loop trajectory: smooth periodic eased sinusoidal trajectory to prevent jumps at loop boundary
        if motion_intent.loop:
            # t progresses smoothly as a sine wave that starts and ends at 0.0
            # loop trajectory must span a full loop (t starts at 0, goes to 1, then back to 0)
            t_val = 0.5 - 0.5 * np.cos(2.0 * np.pi * float(i) / num_frames)
        else:
            t_val = float(i) / (num_frames - 1) if num_frames > 1 else 0.0

        requested_pos[2] += push_distance * t_val

        # Call the actual production renderer!
        rendered_image = render_view(
            resized_reconstructed_slices,
            camera_matrix,
            card_corners_3d_list,
            requested_pos,
            original_size=(height, width),
            original_slices=resized_original_slices,
            parallax_strength=motion_plan.parallax_strength,
            camera_motion_strength=motion_plan.camera_motion_strength,
            zoom_strength=motion_plan.zoom_strength,
            rotation_strength=motion_plan.rotation_strength,
            start_camera_position=start_camera_position,
            t=t_val,
        )

        frame_name = f"preview_frame_{i:03d}.png"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGR))

        # Resolve path relative to serve directory
        rel_path = f"/{AppState.SRV_DIR}/{state.filename}/{frame_name}"
        preview_frame_paths.append(rel_path)

        if progress_callback:
            progress_callback(i + 1, num_frames)

    return PreviewResult(
        frames=preview_frame_paths,
        fps=24,
        duration=float(num_frames) / 24.0,
        quality=motion_plan.num_frames, # mapped to quality mode
        motion_plan_id=str(hash(motion_plan)),
        diagnostics={"motion_plan": motion_plan.to_dict()}
    )
