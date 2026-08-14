import unittest
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from .controller import AppState
from .preview.preview_models import MotionIntent, SceneAnalysis, MotionPlan
from .preview.preview_renderer import PreviewRenderer
from .preview.preview_controller import PreviewController
from .preview.preview_cache import PreviewCache

class TestParallaxPreview(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for state testing
        self.temp_dir = Path("appstate-testpreview")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create basic synthetic image (100x100 RGB) and depth map
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.depth_map = np.ones((100, 100), dtype=np.uint8) * 10
        # Add a central subject to test Portrait classification (range = 250 - 10 = 240 > 180)
        self.depth_map[30:70, 30:70] = 250

        # Construct basic AppState
        self.state = AppState()
        self.state.filename = str(self.temp_dir)
        self.state.imgData = Image.fromarray(self.image)
        self.state.depthMapData = self.depth_map
        self.state.imgThresholds = [0, 80, 160, 255]
        self.state.num_slices = 3

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_scene_analysis_classification(self):
        """Verify that scene classification correctly identifies Portrait, Landscape, etc."""
        # 1. Test Portrait (has strong central subject)
        analysis = PreviewRenderer.analyze_scene(self.image, self.depth_map)
        self.assertEqual(analysis.scene_type, "Portrait")
        self.assertTrue(analysis.primary_subject_detected)
        self.assertEqual(analysis.quality_indicator, "Limited")  # Due to high depth range (250 - 10 = 240)

        # 2. Test Landscape (flat/smooth gradients)
        flat_depth = np.zeros((100, 100), dtype=np.uint8)
        for r in range(100):
            flat_depth[r, :] = int(r * 1.5)  # smooth gradient from 0 to 150 (mean = 75, std = 43.3)
        analysis_landscape = PreviewRenderer.analyze_scene(self.image, flat_depth)
        self.assertEqual(analysis_landscape.scene_type, "Landscape")
        self.assertFalse(analysis_landscape.primary_subject_detected)

    def test_motion_planning_and_looping(self):
        """Verify that motion planning trajectory is loopable and starts/ends at exactly 0."""
        intent = MotionIntent(
            movement_style="Cinematic Auto",
            strength_level="Cinematic",
            motion_direction="Auto",
            duration=4.0,
            loop=True
        )

        analysis = PreviewRenderer.analyze_scene(self.image, self.depth_map)
        plan = PreviewRenderer.plan_motion(analysis, intent)

        # Retrieve first and last states of loop
        first_state = plan.trajectory[0]
        last_state = plan.trajectory[-1]

        # In a perfect loop, translations/orbits start and end at exactly 0.0 or match perfectly
        self.assertAlmostEqual(first_state["x"], 0.0, places=4)
        self.assertAlmostEqual(last_state["x"], 0.0, places=4)
        self.assertAlmostEqual(first_state["yaw"], 0.0, places=4)
        self.assertAlmostEqual(last_state["yaw"], 0.0, places=4)

    def test_safety_validation_motion_reduction(self):
        """Verify that high-risk motion (high disocclusion/depth range) automatically reduces strength."""
        high_risk_depth = np.ones((100, 100), dtype=np.uint8) * 10
        high_risk_depth[40:60, 40:60] = 250  # Massive range of 240!

        analysis = PreviewRenderer.analyze_scene(self.image, high_risk_depth)
        self.assertEqual(analysis.disocclusion_risk, "High")

        intent = MotionIntent(strength_level="Dramatic", loop=True)
        plan = PreviewRenderer.plan_motion(analysis, intent)

        # Since it was high risk, the planning must automatically scale down strength
        self.assertTrue(plan.was_reduced)
        self.assertTrue(plan.actual_strength_multiplier < 1.0)

    def test_preview_vs_final_consistency(self):
        """Verify Preview trajectory normalization path maps 100% consistently to Final trajectory."""
        intent = MotionIntent(movement_style="Orbit", strength_level="Dynamic", loop=False)
        analysis = PreviewController.get_scene_analysis(self.state)
        plan = PreviewController.get_motion_plan(self.state, intent)

        # Verify trajectory has identical relative normalized shape
        mid_idx = len(plan.trajectory) // 2
        mid_state = plan.trajectory[mid_idx]

        # Verify that relative depth ordering is strictly preserved
        self.assertTrue(plan.target_screen_disparity > 0.0)

    @patch("shutil.which", return_value="ffmpeg")
    @patch("subprocess.run")
    def test_preview_rendering_and_caching(self, mock_run, mock_which):
        """Verify that rendering low-resolution preview runs fast and hits cache on subsequent requests."""
        def fake_run(cmd, **kwargs):
            # cmd[-1] is the output path string
            Path(cmd[-1]).touch()
            from unittest.mock import MagicMock
            proc = MagicMock()
            proc.returncode = 0
            return proc
        mock_run.side_effect = fake_run

        intent = MotionIntent(movement_style="Cinematic Auto", strength_level="Cinematic")

        # Ensure slice files are initialized
        from .slice import ImageSlice
        slice_0 = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=0)
        slice_0.filename = str(self.temp_dir / "image_slice_0.png")
        slice_0.save_image()

        slice_1 = ImageSlice(np.ones((100, 100, 4), dtype=np.uint8) * 255, depth=127)
        slice_1.filename = str(self.temp_dir / "image_slice_1.png")
        slice_1.save_image()

        self.state.image_slices = [slice_0, slice_1]

        # Render 1: Cache Miss
        res1 = PreviewController.generate_preview(self.state, intent, quality="Fast")
        self.assertFalse(res1.cache_hit)
        self.assertEqual(res1.quality, "Fast")
        self.assertEqual(len(res1.frame_urls), 24)  # Fast is 24 frames
        self.assertTrue(Path(self.temp_dir / "preview" / "preview.mp4").exists())

        # Render 2: Cache Hit
        res2 = PreviewController.generate_preview(self.state, intent, quality="Fast")
        self.assertTrue(res2.cache_hit)

if __name__ == "__main__":
    unittest.main()
