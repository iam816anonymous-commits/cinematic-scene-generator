import os
import unittest
import numpy as np
from pathlib import Path

from .slice import ImageSlice
from .camera import Camera
from .segmentation import render_view, render_image_sequence

class TestReferenceViewSynthesis(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(100.0, 500.0, 100.0)
        self.camera_matrix = self.camera.camera_matrix(100, 100)

        self.bg_slice = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=0)
        self.mg_slice = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=10)
        self.fg_slice = ImageSlice(np.ones((120, 120, 4), dtype=np.uint8) * 255, depth=20)
        self.image_slices = [self.bg_slice, self.mg_slice, self.fg_slice]

        self.bg_orig = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=0)
        self.mg_orig = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=10)
        self.fg_orig = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=20)
        self.original_slices = [self.bg_orig, self.mg_orig, self.fg_orig]

        self.card_corners = [
            self.bg_slice.create_card(100, 100, self.camera),
            self.mg_slice.create_card(100, 100, self.camera),
            self.fg_slice.create_card(100, 100, self.camera),
        ]
        for card in self.card_corners:
            card[:, :2] *= 1.2

    def test_immutable_reference_and_provenance(self):
        """Verify immutable reference source, no recursive warping, source pixel preservation, and provenance tracking."""
        start_pos = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        cam_pos = np.array([5.0, 2.0, -95.0], dtype=np.float32)

        # Render view
        rendered = render_view(
            self.image_slices, self.camera_matrix, self.card_corners, cam_pos,
            original_size=(100, 100), original_slices=self.original_slices,
            parallax_strength=1.0, start_camera_position=start_pos,
            perceptual_parallax_strength=0.70, t=0.5
        )

        # Check provenance mapping
        prov_map = rendered.provenance_map
        unique_prov = np.unique(prov_map)

        # Verify that all tracked provenance labels are correct
        for label in unique_prov:
            self.assertTrue(label in [0, 1, 2, 3, 4])

        # Recursive reconstruction must be exactly 0.0
        self.assertEqual(rendered.recursively_reconstructed_percentage, 0.0)

        # Source pixel preservation should hold (original pixels are present)
        self.assertTrue(1 in unique_prov or 2 in unique_prov)

        # Check fidelity score is valid
        self.assertTrue(0.0 <= rendered.reference_fidelity_score <= 100.0)

    def test_disparity_driven_pixel_displacement(self):
        """Verify that screen space disparity is correctly scaled with perceptual strength."""
        start_pos = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        cam_pos = np.array([0.0, 0.0, -100.0], dtype=np.float32)

        # Render 0.55 strength
        rendered_55 = render_view(
            self.image_slices, self.camera_matrix, self.card_corners, cam_pos,
            original_size=(100, 100), original_slices=self.original_slices,
            parallax_strength=1.0, start_camera_position=start_pos,
            perceptual_parallax_strength=0.55, t=0.5
        )

        # Render 0.70 strength
        rendered_70 = render_view(
            self.image_slices, self.camera_matrix, self.card_corners, cam_pos,
            original_size=(100, 100), original_slices=self.original_slices,
            parallax_strength=1.0, start_camera_position=start_pos,
            perceptual_parallax_strength=0.70, t=0.5
        )

        # Disparity must grow monotonically with strength
        self.assertTrue(rendered_70.screen_space_max_disparity_px > rendered_55.screen_space_max_disparity_px)

if __name__ == "__main__":
    unittest.main()
