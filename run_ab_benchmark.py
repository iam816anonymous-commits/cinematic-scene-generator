import os
import sys
import time
import resource
import numpy as np
import cv2
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path(__file__).parent))

import torch
from parallax_maker.slice import ImageSlice
from parallax_maker.camera import Camera
from parallax_maker.segmentation import (
    analyze_depth_histogram,
    generate_image_slices,
    render_image_sequence,
    reconstruct_slice_disocclusions,
    render_view
)
from parallax_maker.motion import CinematicMotionModel

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_benchmark():
    print("="*60)
    print("      PARALLAX MAKER A/B BENCHMARK RUNNER")
    print("="*60)

    # Load input image and depth map
    img_path = Path("example/input.png")
    depth_path = Path("example/depth_map.png")

    if not img_path.exists() or not depth_path.exists():
        print("Error: benchmark assets not found!")
        return

    # 1. Preprocessing Stage (identical for both)
    t_pre_start = time.perf_counter()
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    h_orig, w_orig = image.shape[:2]

    num_slices = 5
    thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)
    original_slices = generate_image_slices(image, depth_map, thresholds, num_expand=0)

    margin = 0.1
    reconstructed_slices = []
    for i, slice_image in enumerate(original_slices):
        recon_img = reconstruct_slice_disocclusions(slice_image.image, is_background=(i == 0), margin=margin)
        recon_slice = ImageSlice(image=recon_img, depth=slice_image.depth)
        reconstructed_slices.append(recon_slice)

    t_pre_end = time.perf_counter()
    preprocessing_time = t_pre_end - t_pre_start

    # Camera Setup
    camera = Camera(100.0, 500.0, 100.0)
    camera_matrix = camera.camera_matrix(w_orig, h_orig)

    card_corners_3d_list = []
    for i, image_slice in enumerate(reconstructed_slices):
        card = image_slice.create_card(h_orig, w_orig, camera)
        card[:, :2] *= (1.0 + 2.0 * margin)
        card_corners_3d_list.append(card)

    # --------------------------------------------------
    # RUN SYSTEM A: Baseline Renderer
    # --------------------------------------------------
    print("\n[System A] Running Baseline Renderer (300 frames)...")
    out_dir_a = Path("/tmp/rendered_output_ab_a")
    if out_dir_a.exists():
        import shutil
        shutil.rmtree(out_dir_a)
    out_dir_a.mkdir(parents=True, exist_ok=True)

    ram_start_a = get_peak_ram_mb()
    t_render_start_a = time.perf_counter()

    # Large physical camera displacement push-in
    push_distance_a = 150.0
    camera_position_a = np.array([0.0, 0.0, -100.0], dtype=np.float32)

    report_a = render_image_sequence(
        out_dir_a,
        reconstructed_slices,
        card_corners_3d_list,
        camera_matrix,
        camera_position_a,
        push_distance=push_distance_a,
        num_frames=300,
        original_size=(h_orig, w_orig),
        original_slices=original_slices,
        max_reconstruction_ratio=0.15,
        ai_threshold_ratio=0.08,
        motion_mode="baseline"
    )
    t_render_end_a = time.perf_counter()
    render_time_a = t_render_end_a - t_render_start_a
    ram_end_a = get_peak_ram_mb()

    # --------------------------------------------------
    # RUN SYSTEM B: Cinematic Perceived-Depth Parallax
    # --------------------------------------------------
    print("\n[System B] Running Cinematic Motion Renderer (300 frames)...")
    out_dir_b = Path("/tmp/rendered_output_ab_b")
    if out_dir_b.exists():
        import shutil
        shutil.rmtree(out_dir_b)
    out_dir_b.mkdir(parents=True, exist_ok=True)

    ram_start_b = get_peak_ram_mb()
    t_render_start_b = time.perf_counter()

    # Bounded camera movement with depth weights and automatic scene-adapted limits
    push_distance_b = 150.0
    camera_position_b = np.array([0.0, 0.0, -100.0], dtype=np.float32)
    cinematic_profile = CinematicMotionModel.analyze_scene_and_build_profile(depth_map)
    print(f"  Selected Profile: {cinematic_profile}")

    report_b = render_image_sequence(
        out_dir_b,
        reconstructed_slices,
        card_corners_3d_list,
        camera_matrix,
        camera_position_b,
        push_distance=push_distance_b,
        num_frames=300,
        original_size=(h_orig, w_orig),
        original_slices=original_slices,
        max_reconstruction_ratio=0.15,
        ai_threshold_ratio=0.08,
        motion_mode="cinematic",
        cinematic_profile=cinematic_profile
    )
    t_render_end_b = time.perf_counter()
    render_time_b = t_render_end_b - t_render_start_b
    ram_end_b = get_peak_ram_mb()

    # Peak VRAM if available
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --------------------------------------------------
    # PRINT SIDE-BY-SIDE COMPARISON REPORT
    # --------------------------------------------------
    print("\n" + "="*70)
    print("               A/B BENCHMARK COMPARATIVE REPORT")
    print("="*70)
    print(f"{'Metric':<35} | {'System A (Baseline)':<15} | {'System B (Cinematic)':<15}")
    print("-"*70)
    print(f"{'Preprocessing Time':<35} | {preprocessing_time:13.4f}s | {preprocessing_time:13.4f}s")
    print(f"{'Rendering Time (300 frames)':<35} | {render_time_a:13.4f}s | {render_time_b:13.4f}s")
    print(f"{'Total Execution Time':<35} | {preprocessing_time+render_time_a:13.4f}s | {preprocessing_time+render_time_b:13.4f}s")
    print(f"{'Peak RAM Usage':<35} | {ram_end_a:11.2f} MB | {ram_end_b:11.2f} MB")
    print(f"{'Peak VRAM Usage':<35} | {peak_vram_mb:11.2f} MB | {peak_vram_mb:11.2f} MB")
    print(f"{'Max Reconstructed Area ratio':<35} | {report_a['max_reconstructed_ratio']*100:11.2f}% | {report_b['max_reconstructed_ratio']*100:11.2f}%")
    print(f"{'Avg Reconstructed Area ratio':<35} | {report_a['average_reconstructed_ratio']*100:11.2f}% | {report_b['average_reconstructed_ratio']*100:11.2f}%")
    print(f"{'Mean Temporal MAD (Flicker)':<35} | {report_a['mean_temporal_mad']:13.4f} | {report_b['mean_temporal_mad']:13.4f}")
    print(f"{'Max Temporal MAD (Popping)':<35} | {report_a['max_temporal_mad']:13.4f} | {report_b['max_temporal_mad']:13.4f}")
    print(f"{'Mean Mask Change Ratio':<35} | {report_a['mean_mask_change']:13.4f} | {report_b['mean_mask_change']:13.4f}")
    print(f"{'Mean Boundary Movement':<35} | {report_a['mean_boundary_movement']:13.4f} | {report_b['mean_boundary_movement']:13.4f}")
    print(f"{'AI Reconstruction Frames':<35} | {report_a['ai_used_count']:11d}/300 | {report_b['ai_used_count']:11d}/300")
    print(f"{'Clamped Frames percentage':<35} | {report_a['percentage_clamped_frames']:11.2f}% | {report_b['percentage_clamped_frames']:11.2f}%")
    print(f"{'Reconstruction Budget Warnings':<35} | {len(report_a['warnings']):13d} | {len(report_b['warnings']):13d}")
    print("="*70)

    # --------------------------------------------------
    # VISUAL CHECKPOINTS ANALYSIS
    # --------------------------------------------------
    print("\nVisual Checkpoint Metrics for Frame Checkpoints (0, 25, 75, 150, 225, 299):")
    print("-"*70)
    print(f"{'Frame':<8} | {'System A (Baseline) Recon %':<28} | {'System B (Cinematic) Recon %':<28}")
    print("-"*70)
    for f in [0, 25, 75, 150, 225, 299]:
        # Compute ratio A
        req_pos_a = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        req_pos_a[2] += (push_distance_a / 300.0) * f
        rendered_a = render_view(
            reconstructed_slices, camera_matrix, card_corners_3d_list, req_pos_a,
            original_size=(w_orig, h_orig), original_slices=original_slices, motion_mode="baseline"
        )
        # Compute ratio B
        rendered_b = render_view(
            reconstructed_slices, camera_matrix, card_corners_3d_list, req_pos_a,
            original_size=(w_orig, h_orig), original_slices=original_slices,
            motion_mode="cinematic", cinematic_profile=cinematic_profile, frame_idx=f, total_frames=300
        )

        # Provenance breakdowns
        p_map_a = rendered_a.provenance_map
        p_map_b = rendered_b.provenance_map

        recon_a_pct = rendered_a.reconstruction_ratio * 100.0
        recon_b_pct = rendered_b.reconstruction_ratio * 100.0

        print(f"{f:5d}    | {recon_a_pct:24.2f}% | {recon_b_pct:24.2f}%")
        print(f"         | Provenance A: Orig={(np.count_nonzero(p_map_a==1)/p_map_a.size)*100:.1f}%, Det={(np.count_nonzero(p_map_a==2)/p_map_a.size)*100:.1f}%, AI={(np.count_nonzero(p_map_a==3)/p_map_a.size)*100:.1f}%")
        print(f"         | Provenance B: Orig={(np.count_nonzero(p_map_b==1)/p_map_b.size)*100:.1f}%, Det={(np.count_nonzero(p_map_b==2)/p_map_b.size)*100:.1f}%, AI={(np.count_nonzero(p_map_b==3)/p_map_b.size)*100:.1f}%")
        print("-"*70)

if __name__ == "__main__":
    run_benchmark()
