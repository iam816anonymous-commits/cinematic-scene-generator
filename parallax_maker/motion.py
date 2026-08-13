# (c) 2026 Niels Provos & Jules

import numpy as np
import cv2

class MotionProfile:
    """
    Encapsulates cinematic camera motion profile parameters for perceived-depth parallax.
    """
    def __init__(
        self,
        max_displacement_x=12.0,
        max_displacement_y=4.0,
        max_displacement_z=6.0,
        depth_response="sigmoid",
        zoom_amount=1.02,
        safety_factor=1.0,
        reconstruction_budget=0.15,
        profile_name="Standard"
    ):
        self.max_displacement_x = max_displacement_x
        self.max_displacement_y = max_displacement_y
        self.max_displacement_z = max_displacement_z
        self.depth_response = depth_response  # "linear", "exponential", "sigmoid"
        self.zoom_amount = zoom_amount
        self.safety_factor = safety_factor
        self.reconstruction_budget = reconstruction_budget
        self.profile_name = profile_name

    def __str__(self):
        return (
            f"MotionProfile(name={self.profile_name}, max_dx={self.max_displacement_x:.2f}, "
            f"max_dy={self.max_displacement_y:.2f}, max_dz={self.max_displacement_z:.2f}, "
            f"response={self.depth_response}, zoom={self.zoom_amount:.3f}, safety={self.safety_factor:.2f})"
        )


class CinematicMotionModel:
    """
    Decoupled cinematic motion abstraction that receives scene depth and parameters,
    performs scene-adaptive analysis, and computes depth-weighted differential displacements.
    """
    @staticmethod
    def analyze_scene_and_build_profile(depth_map):
        """
        Inspects the generated depth map and derives characteristics to automatically
        select a highly conservative, artifact-free cinematic motion profile.
        """
        if depth_map is None or depth_map.size == 0:
            return MotionProfile()

        # Measure spatial depth characteristics
        min_depth = float(depth_map.min())
        max_depth = float(depth_map.max())
        depth_range = max_depth - min_depth
        depth_variance = float(np.var(depth_map))
        depth_mean = float(np.mean(depth_map))

        # Calculate region occupancies
        foreground_occupancy = float(np.mean(depth_map > 170))
        background_occupancy = float(np.mean(depth_map < 85))

        # Calculate edge density and depth gradients (detecting high disocclusion risk)
        sobelx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        depth_gradients = float(np.mean(np.abs(sobelx) + np.abs(sobely)))

        canny = cv2.Canny(depth_map, 50, 150)
        edge_density = float(np.mean(canny > 0))

        # Base default parameters
        base_dx = 12.0
        base_dy = 4.0
        base_dz = 6.0
        zoom = 1.02
        reconstruction_budget = 0.15
        depth_response = "sigmoid"
        profile_name = "Standard"

        # 1. Automatic Scene Profiling & Classification
        if foreground_occupancy > 0.35 and depth_variance > 1000:
            # Portrait / Close-up: Shallow parallax, stronger subject separation, capped vertical drift
            base_dx = 6.0
            base_dy = 2.0
            base_dz = 3.0
            zoom = 1.012
            depth_response = "sigmoid"
            profile_name = "Portrait"
        elif background_occupancy > 0.45 and depth_variance < 800:
            # Landscape / Wide-angle: Larger but still bounded landscape panning
            base_dx = 16.0
            base_dy = 6.0
            base_dz = 10.0
            zoom = 1.03
            depth_response = "exponential"
            profile_name = "Landscape"

        # 2. Risk Adjustment & Budget Enforcement (Reduce motion amplitude before rendering)
        safety_factor = 1.0
        if edge_density > 0.04:
            # High edge complexity -> reduce motion to avoid dynamic flickering
            safety_factor *= 0.70
        if depth_gradients > 8.0:
            # Large depth discontinuities -> scale down displacement
            safety_factor *= 0.75
        if depth_range < 50.0:
            # Noisy, shallow, or uncertain depth map -> extremely conservative displacement
            safety_factor *= 0.60

        profile = MotionProfile(
            max_displacement_x=base_dx * safety_factor,
            max_displacement_y=base_dy * safety_factor,
            max_displacement_z=base_dz * safety_factor,
            depth_response=depth_response,
            zoom_amount=zoom,
            safety_factor=safety_factor,
            reconstruction_budget=reconstruction_budget,
            profile_name=profile_name
        )
        return profile

    @staticmethod
    def get_displacement_weights(slice_depth, min_depth, max_depth, response_type="sigmoid"):
        """
        Computes a normalized motion response weight based on a configurable response curve.
        """
        d_range = float(max_depth - min_depth)
        if d_range <= 0.0:
            return 0.5

        # Normalize depth to [0.0, 1.0] where 0 is background (far) and 1 is foreground (near)
        normalized_depth = float(slice_depth - min_depth) / d_range
        normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

        if response_type == "sigmoid":
            # Sharp subject separation / s-curve
            return float(1.0 / (1.0 + np.exp(-10.0 * (normalized_depth - 0.5))))
        elif response_type == "exponential":
            # Smooth ramping / exponential curve
            return float(normalized_depth ** 2)
        else:
            # Standard linear mapping
            return normalized_depth
