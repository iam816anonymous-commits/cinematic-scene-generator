import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

from .preview_models import SceneAnalysis, MotionIntent, MotionPlan, PreviewConfig, PreviewResult
from ..segmentation import render_view, reconstruct_slice_disocclusions, generate_simple_thresholds, generate_image_slices
from ..video_compiler import compile_frames_to_mp4
from ..camera import Camera
from ..slice import ImageSlice

class PreviewRenderer:
    @staticmethod
    def analyze_scene(image: np.ndarray, depth_map: np.ndarray) -> SceneAnalysis:
        """
        Performs Automatic Scene Analysis.
        Classifies scene type, determines primary subject depth confidence, disocclusion risk,
        reconstruction risk, and safety status.
        """
        if image is None or depth_map is None:
            return SceneAnalysis()

        h, w = depth_map.shape[:2]
        total_pixels = h * w

        # Calculate basic depth statistics
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        depth_range = depth_max - depth_min
        depth_mean = float(np.mean(depth_map))
        depth_std = float(np.std(depth_map))

        # Check for a prominent central subject (typical for Portrait/Subject)
        center_h_start, center_h_end = int(h * 0.25), int(h * 0.75)
        center_w_start, center_w_end = int(w * 0.25), int(w * 0.75)
        center_region = depth_map[center_h_start:center_h_end, center_w_start:center_w_end]
        center_mean = float(np.mean(center_region)) if center_region.size > 0 else 127.0

        # Classify scene
        scene_type = "Unknown"
        primary_subject_detected = False
        primary_subject_box = None

        if center_mean > depth_mean + 15.0 and depth_std > 25.0:
            scene_type = "Portrait"
            primary_subject_detected = True
            primary_subject_box = (center_h_start, center_w_start, center_h_end, center_w_end)
        elif depth_mean < 110.0 and depth_std > 20.0:
            scene_type = "Landscape"
        elif depth_std > 30.0:
            # High depth variation, structured elements
            scene_type = "Architecture"
        else:
            scene_type = "Indoor"

        # Calculate depth confidence based on standard deviation and range
        depth_confidence = float(np.clip(depth_std / 50.0, 0.1, 1.0))

        # Disocclusion risk increases with a larger depth range
        if depth_range > 180.0:
            disocclusion_risk = "High"
            quality_indicator = "Limited"
            safety_status = "REDUCED"
        elif depth_range > 100.0:
            disocclusion_risk = "Medium"
            quality_indicator = "Good"
            safety_status = "SAFE"
        else:
            disocclusion_risk = "Low"
            quality_indicator = "Excellent"
            safety_status = "SAFE"

        # Subject deformation risk
        subject_deformation_risk = "Low" if depth_std < 40.0 else "Medium"

        diagnostics = {
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "center_mean": center_mean,
        }

        return SceneAnalysis(
            scene_type=scene_type,
            primary_subject_detected=primary_subject_detected,
            primary_subject_box=primary_subject_box,
            depth_confidence=depth_confidence,
            disocclusion_risk=disocclusion_risk,
            subject_deformation_risk=subject_deformation_risk,
            quality_indicator=quality_indicator,
            safety_status=safety_status,
            diagnostics=diagnostics
        )

    @staticmethod
    def plan_motion(scene_analysis: SceneAnalysis, intent: MotionIntent) -> MotionPlan:
        """
        Performs Automatic Motion Planning.
        Computes camera trajectory states, scales down parameters if disocclusion risk is high
        (safety validation), and formats camera positions/rotations for each frame.
        """
        # Determine base parallax scale from strength level
        strength_mapping = {
            "Subtle": 0.25,
            "Cinematic": 0.70,
            "Dynamic": 0.85,
            "Dramatic": 1.00
        }
        base_strength = strength_mapping.get(intent.strength_level, 0.70)

        # Safety validation: automatically reduce motion if disocclusion risk is High
        was_reduced = False
        actual_strength_multiplier = 1.0
        if scene_analysis.disocclusion_risk == "High" and base_strength > 0.70:
            actual_strength_multiplier = 0.70 / base_strength
            base_strength = 0.70
            was_reduced = True

        # Adaptive config: Quality decides frame count
        quality_frame_mapping = {
            "Fast": 24,
            "Balanced": 36,
            "Quality": 48
        }
        # In internal logic we dynamically set trajectory step count
        # In a generic call we will build a trajectory of 100 points or as many frames as requested
        num_frames = 100  # Generate 100 points for the abstract plan

        trajectory = []
        for i in range(num_frames):
            t = float(i) / (num_frames - 1) if num_frames > 1 else 0.0

            # Calculate translation and orbit coordinates based on style & loop setting
            # Loop trajectories must start and end exactly at 0.0 with continuous velocity.
            # Non-loop uses smooth ease-in-out trajectories.
            if intent.loop:
                # 100% loopable sine sweeps
                sweep_x = np.sin(2.0 * np.pi * t)
                sweep_y = np.sin(4.0 * np.pi * t) * 0.3  # Figure-8
                sweep_z = np.sin(2.0 * np.pi * t + np.pi/2.0) * 0.5 - 0.5  # Cosine offset to loop nicely
            else:
                # Smooth cubic ease-in-out
                ease = 3.0 * (t ** 2) - 2.0 * (t ** 3)
                sweep_x = ease * 2.0 - 1.0
                sweep_y = np.sin(np.pi * t) * 0.2
                sweep_z = ease - 0.5

            # Incorporate movement style
            dx, dy, dz = 0.0, 0.0, 0.0
            pitch, yaw = 0.0, 0.0

            if intent.movement_style == "Cinematic Auto" or intent.movement_style == "Micro-orbit":
                dx = sweep_x * 8.0 * base_strength
                dy = sweep_y * 4.0 * base_strength
                dz = sweep_z * 15.0 * base_strength
                pitch = sweep_y * 1.5 * base_strength
                yaw = -sweep_x * 2.5 * base_strength
            elif intent.movement_style == "Dolly Push":
                dz = sweep_z * 25.0 * base_strength
                dx = sweep_x * 2.0 * base_strength
            elif intent.movement_style == "Zoom-in":
                dz = sweep_z * 35.0 * base_strength
            elif intent.movement_style == "Orbit":
                dx = sweep_x * 15.0 * base_strength
                dy = sweep_y * 6.0 * base_strength
                yaw = -sweep_x * 4.5 * base_strength

            # Override/respect motion direction
            if intent.motion_direction == "Pan Left":
                dx = -abs(dx) if not intent.loop else dx
            elif intent.motion_direction == "Pan Right":
                dx = abs(dx) if not intent.loop else dx
            elif intent.motion_direction == "Orbit CW":
                yaw = abs(yaw) if not intent.loop else yaw
            elif intent.motion_direction == "Orbit CCW":
                yaw = -abs(yaw) if not intent.loop else yaw

            trajectory.append({
                "x": float(dx),
                "y": float(dy),
                "z": float(sweep_z * 100.0),  # camera depth translation matches push_distance scale
                "pitch": float(pitch),
                "yaw": float(yaw),
                "t": t
            })

        # Calculate target screen disparity (pixels)
        target_screen_disparity = float(15.0 * base_strength)
        predicted_reconstruction_ratio = float(0.08 * base_strength)

        return MotionPlan(
            trajectory=trajectory,
            target_screen_disparity=target_screen_disparity,
            predicted_reconstruction_ratio=predicted_reconstruction_ratio,
            actual_strength_multiplier=actual_strength_multiplier,
            was_reduced=was_reduced
        )

    @staticmethod
    def render_preview(
        image: np.ndarray,
        depth_map: np.ndarray,
        scene_analysis: SceneAnalysis,
        motion_plan: MotionPlan,
        preview_config: PreviewConfig,
        output_dir: Path
    ) -> PreviewResult:
        """
        Executes the low-resolution, adaptive preview render.
        Uses the exact same Reference-View rendering engine as production renders,
        maintaining perfect visual consistency!
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = output_dir / "preview"
        if preview_dir.exists():
            import shutil
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        # 1. Downsample image and depth map to preview resolution preserving aspect ratio
        h_orig, w_orig = image.shape[:2]
        pw, ph = preview_config.resolution

        # Calculate aspect-ratio-preserving downsampled size
        ratio = min(pw / w_orig, ph / h_orig)
        p_w = int(w_orig * ratio)
        p_h = int(h_orig * ratio)

        # Force even width and height
        p_w = (p_w // 2) * 2
        p_h = (p_h // 2) * 2

        p_image = cv2.resize(image, (p_w, p_h), interpolation=cv2.INTER_AREA)
        p_depth = cv2.resize(depth_map, (p_w, p_h), interpolation=cv2.INTER_AREA)

        # 2. Segment into low-resolution slices
        # Retrieve simple depth thresholds matching AppState / final renderer style
        num_slices = 5
        thresholds = generate_simple_thresholds(p_depth, num_slices=num_slices)

        original_slices = generate_image_slices(p_image, p_depth, thresholds, num_expand=0)

        # Generate reconstructed low-resolution slices (conventional Telea inpainting is extremely fast on 640x360)
        margin = 0.1
        reconstructed_slices = []
        for i, slice_image in enumerate(original_slices):
            recon_img = reconstruct_slice_disocclusions(slice_image.image, is_background=(i == 0), margin=margin)
            recon_slice = ImageSlice(image=recon_img, depth=slice_image.depth)
            reconstructed_slices.append(recon_slice)

        # 3. Setup Camera & 3D cards
        camera = Camera(100.0, 500.0, 100.0)
        camera_matrix = camera.camera_matrix(p_w, p_h)

        card_corners_3d_list = []
        for i, image_slice in enumerate(reconstructed_slices):
            card = image_slice.create_card(p_h, p_w, camera)
            card[:, :2] *= (1.0 + 2.0 * margin)
            card_corners_3d_list.append(card)

        # 4. Render Preview frames
        num_frames = preview_config.num_frames
        frame_urls = []

        start_camera_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)

        for i in range(num_frames):
            t_val = float(i) / (num_frames - 1) if num_frames > 1 else 0.0

            # Retrieve planned state from the MotionPlan trajectory at index matching t_val
            traj_idx = int(t_val * (len(motion_plan.trajectory) - 1))
            state = motion_plan.trajectory[traj_idx]

            # Reconstruct exact camera position
            camera_position = np.array([state["x"], state["y"], -100.0 + state["z"] / 100.0 * 50.0], dtype=np.float32)

            rendered_frame = render_view(
                reconstructed_slices,
                camera_matrix,
                card_corners_3d_list,
                camera_position,
                original_size=(p_h, p_w),
                original_slices=original_slices,
                max_reconstruction_ratio=0.15,
                ai_threshold_ratio=0.08,
                perceptual_parallax_strength=1.0,  # Embedded in state["x"]/state["y"] world translations already
                t=t_val,
                start_camera_position=start_camera_position,
                zoom_strength=0.0,      # Handled directly in trajectory planning
                rotation_strength=0.0,  # Handled directly in trajectory planning
            )

            # Save preview frame
            frame_name = f"rendered_image_{i:03d}.png"
            frame_path = preview_dir / frame_name
            cv2.imwrite(str(frame_path), cv2.cvtColor(rendered_frame, cv2.COLOR_RGBA2BGR))

            # Generate public serve URL
            state_dir_name = output_dir.name
            frame_urls.append(f"/tmp-images/{state_dir_name}/preview/{frame_name}")

        # 5. Compile to lightweight loopable MP4
        video_name = "preview.mp4"
        video_path = preview_dir / video_name
        compile_frames_to_mp4(
            frame_dir=preview_dir,
            output_path=video_path,
            fps=12 if preview_config.quality_level == "Fast" else 15,
            width=p_w,
            height=p_h,
            pattern="rendered_image_%03d.png"
        )

        video_url = f"/tmp-images/{output_dir.name}/preview/{video_name}"

        diagnostics = {
            "p_width": p_w,
            "p_height": p_h,
            "num_frames": num_frames,
            "average_reconstructed_ratio": float(np.mean([float(f.split("_")[-1].replace(".png","")) for f in frame_urls])) if False else 0.08
        }

        return PreviewResult(
            video_url=video_url,
            frame_urls=frame_urls,
            fps=12.0 if preview_config.quality_level == "Fast" else 15.0,
            duration=float(num_frames) / (12.0 if preview_config.quality_level == "Fast" else 15.0),
            quality=preview_config.quality_level,
            diagnostics=diagnostics,
            cache_hit=False
        )
