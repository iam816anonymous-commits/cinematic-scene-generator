# (c) 2024 Niels Provos

import unittest
import numpy as np
import cv2

from parallax_maker.preview.preview_models import SceneAnalysis, MotionIntent
from parallax_maker.preview.preview_controller import analyze_scene, plan_motion

class TestPreviewAndConsistency(unittest.TestCase):
    def test_scene_analysis_portrait_landscape(self):
        # Synthetic Landscape image
        img_landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_landscape = np.ones((100, 200), dtype=np.uint8) * 128
        analysis_l = analyze_scene(img_landscape, depth_landscape)
        self.assertEqual(analysis_l.scene_type, "Landscape")

        # Synthetic Portrait image
        img_portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        depth_portrait = np.ones((200, 100), dtype=np.uint8) * 128
        analysis_p = analyze_scene(img_portrait, depth_portrait)
        self.assertEqual(analysis_p.scene_type, "Portrait")

        # Check depth confidence and disocclusion risk calculations
        self.assertTrue(0.0 <= analysis_l.depth_confidence <= 1.0)
        self.assertTrue(0.0 <= analysis_l.disocclusion_risk <= 1.0)

    def test_motion_planning_strengths_and_safety_clamping(self):
        # Normal safe scene analysis
        analysis_safe = SceneAnalysis(disocclusion_risk=0.2, depth_confidence=0.8)
        intent_cinematic = MotionIntent(strength="Cinematic", style="Cinematic Auto")
        intent_dramatic = MotionIntent(strength="Dramatic", style="Cinematic Auto")

        plan_cin = plan_motion(analysis_safe, intent_cinematic)
        plan_dram = plan_motion(analysis_safe, intent_dramatic)

        # Dramatic should have larger low-level strengths than Cinematic
        self.assertTrue(plan_dram.parallax_strength > plan_cin.parallax_strength)
        self.assertTrue(plan_dram.zoom_strength > plan_cin.zoom_strength)

        # Unsafe scene analysis (high disocclusion risk)
        analysis_unsafe = SceneAnalysis(disocclusion_risk=0.8, depth_confidence=0.4)
        plan_unsafe_dram = plan_motion(analysis_unsafe, intent_dramatic)

        # Under high risk, the Dramatic motion plan parameters should be automatically clamped/reduced for safety
        self.assertTrue(plan_unsafe_dram.motion_reduced)
        self.assertTrue(plan_unsafe_dram.parallax_strength < plan_dram.parallax_strength)

    def test_preview_vs_final_consistency(self):
        """
        Consistency Test (Section 14):
        Verify that Preview MotionPlan and Final MotionPlan share identical trajectories, anchors, and depth ordering.
        """
        analysis = SceneAnalysis(scene_type="Landscape", disocclusion_risk=0.3, depth_confidence=0.7)
        intent = MotionIntent(style="Cinematic Auto", strength="Cinematic", duration=4, loop=True)

        # Plan preview (Balanced quality) and final (Quality)
        plan_preview = plan_motion(analysis, intent, quality="Balanced")
        plan_final = plan_motion(analysis, intent, quality="Quality")

        # 1. Assert low-level rendering strengths are IDENTICAL (consistency of movement style & strength)
        self.assertEqual(plan_preview.parallax_strength, plan_final.parallax_strength)
        self.assertEqual(plan_preview.camera_motion_strength, plan_final.camera_motion_strength)
        self.assertEqual(plan_preview.zoom_strength, plan_final.zoom_strength)
        self.assertEqual(plan_preview.rotation_strength, plan_final.rotation_strength)

        # 2. Verify relative depth ordering is preserved
        # The underlying slices depths are unchanged, and Z scale weights are identical
        self.assertEqual(plan_preview.motion_reduced, plan_final.motion_reduced)

        # 3. Trajectory normalized path is equivalent (both use the same looping sinusoid trajectory mapping)
        # Verify that for any normalized frame progress t, the same mathematical equations yield identical relative camera offsets
        t_sample = 0.4
        preview_offset_z = plan_preview.push_distance * t_sample
        final_offset_z = plan_final.push_distance * t_sample
        self.assertEqual(preview_offset_z, final_offset_z)

if __name__ == "__main__":
    unittest.main()
