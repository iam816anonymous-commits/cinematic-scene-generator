#!/usr/bin/env python
# (c) 2024 Niels Provos
#
# Uses a depth map to segment an image into multiple slices.
# These slices can be used to create animated 2.5D effects.
#

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
import argparse
from pathlib import Path
from PIL import Image

from .utils import torch_get_device, feather_mask
from .depth import DepthEstimationModel

# for exporting a 3d scene
from .gltf import export_gltf
from .camera import Camera
from .slice import ImageSlice


def generate_depth_map(image, model: DepthEstimationModel, progress_callback=None):
    """
    Generate a depth map from the input image using the specified model.

    Args:
        image (numpy.ndarray): The input image.
        model (DepthEstimationModel): The depth estimation model to use.
            Supported models are "midas" and "dinov2".
        progress_callback (callable, optional): A callback function to report progress.

    Returns:
        numpy.ndarray: The grayscale depth map.
    Raises:
        ValueError: If an unknown model is specified.
    """

    depth_map = model.depth_map(image, progress_callback=progress_callback)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map


def analyze_depth_histogram(depth_map, num_slices=5):
    """Analyze the histogram of the depth map and determine thresholds for segmentation."""

    def calculate_thresholds(depth_map, num_slices):
        thresholds = [0]
        hist, _ = np.histogram(depth_map.flatten(), 256, [0, 256])
        total_pixels = depth_map.shape[0] * depth_map.shape[1]
        target_pixels_per_slice = float(total_pixels) / (num_slices + 1)
        total_sum = 0
        for i in range(1, 256):
            total_sum += hist[i]
            if total_sum >= target_pixels_per_slice * len(thresholds) or i == 255:
                thresholds.append(i)
        return thresholds

    # this is a terrible hack to make sure we get the right number of thresholds
    thresholds = calculate_thresholds(depth_map, num_slices - 1)
    if len(thresholds) != num_slices + 1:
        thresholds = calculate_thresholds(depth_map, num_slices)
    assert (
        len(thresholds) == num_slices + 1
    ), f"Expected {num_slices + 1} thresholds, got {len(thresholds)}"
    return thresholds


def generate_simple_thresholds(depth_map, num_slices=5):
    """Generate simple thresholds based on the depth map."""
    thresholds = [0]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_range = depth_max - depth_min
    step = depth_range / num_slices
    for i in range(num_slices):
        threshold = int(depth_min + (i + 1) * step)
        thresholds.append(threshold)
    return thresholds


def mask_from_depth(depth_map, threshold_min, threshold_max, prev_mask=None):
    """Generate a mask based on the depth map and thresholds."""
    mask = cv2.inRange(depth_map, threshold_min, threshold_max)
    if prev_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(prev_mask))
    return mask


def generate_image_slices(image, depth_map, thresholds, num_expand=50):
    """Generate image slices based on the depth map and thresholds, including an alpha channel.

    Args:
        image (numpy.ndarray): The input image.
        depth_map (numpy.ndarray): The depth map corresponding to the input image.
        thresholds (list): A list of threshold values used to segment the depth map.
        num_expand (int, optional): The number of pixels to expand the mask by. Defaults to 50.

    Returns:
        List[ImageSlice]: A list of image slices.

    """
    slices = []

    prev_mask = None
    for i in range(len(thresholds) - 1):
        threshold_min = thresholds[i]
        threshold_max = thresholds[i + 1]

        mask = mask_from_depth(
            depth_map, threshold_min, threshold_max, prev_mask=prev_mask
        )
        masked_image = create_slice_from_mask(image, mask, num_expand)

        image_slice = ImageSlice(image=masked_image, depth=threshold_max)

        slices.append(image_slice)
        prev_mask = mask

    return slices


def create_slice_from_mask(image, mask, num_expand=50):
    """
    Create a slice from an image based on a given mask.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask to apply on the image.
        num_expand (int): The number of pixels to expand the mask by.

    Returns:
        numpy.ndarray: The masked image slice.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    feathered_mask = feather_mask(mask, num_expand=num_expand)

    # Create a 4-channel image (RGBA)
    masked_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Set alpha channel values based on the feathered mask
    masked_image[:, :, 3] = feathered_mask
    return masked_image


def should_use_ai_fallback(hole_mask, threshold_ratio=0.05):
    """
    Decides whether to use AI inpainting fallback based on the size of the holes.
    """
    if hole_mask is None or hole_mask.size == 0:
        return False
    total_pixels = hole_mask.shape[0] * hole_mask.shape[1]
    hole_pixels = cv2.countNonZero(hole_mask)
    hole_ratio = hole_pixels / total_pixels
    return hole_ratio > threshold_ratio


def reconstruct_slice_disocclusions(image, is_background=False, margin=0.1):
    """
    Reconstructs and pads a slice image to handle disocclusions and prevent black boundaries.

    Args:
        image (numpy.ndarray): RGBA image of the slice.
        is_background (bool): Whether this is the background layer (index 0).
        margin (float): The boundary margin to pad the image.

    Returns:
        numpy.ndarray: The reconstructed and padded RGBA image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    h, w = image.shape[:2]
    pad_h = int(h * margin)
    pad_w = int(w * margin)

    # We pad RGB and Alpha channels
    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)

    # Extract RGB and Alpha
    rgb = padded[:, :, :3].copy()
    alpha = padded[:, :, 3].copy()

    if is_background:
        # For background layer, the entire transparent region (alpha < 255) is a hole
        hole_mask = cv2.inRange(alpha, 0, 254)

        # If there are holes, inpaint them using cv2.inpaint
        if cv2.countNonZero(hole_mask) > 0:
            inpainted_rgb = cv2.inpaint(rgb, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            padded[:, :, :3] = inpainted_rgb

        # Background layer is now completely opaque
        padded[:, :, 3] = 255
    else:
        # For middle/foreground layers, we do edge-padding (dilation)
        # First, find where there is actual content (alpha > 0)
        content_mask = cv2.inRange(alpha, 1, 255)

        # Dilate the content mask slightly outwards
        kernel_size = max(5, int(min(h, w) * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dilated_mask = cv2.dilate(content_mask, kernel, iterations=1)

        # The region to fill is where the dilated mask is active, but the original content is empty
        fill_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(content_mask))

        if cv2.countNonZero(fill_mask) > 0:
            inpainted_rgb = cv2.inpaint(rgb, fill_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            padded[:, :, :3] = inpainted_rgb

        # Dilate the alpha channel as well so that the extended edge fades out smoothly
        dilated_alpha = cv2.dilate(alpha, kernel, iterations=1)
        dilated_alpha_blur = cv2.GaussianBlur(dilated_alpha, (kernel_size, kernel_size), 0)

        # Keep original alpha where it was high, and use the dilated blurred alpha in the extended region
        new_alpha = np.where(alpha > 10, alpha, dilated_alpha_blur)
        padded[:, :, 3] = new_alpha

    return padded


def get_slice_reconstruction_mask(recon_slice_image, orig_slice_image, h_orig, w_orig):
    """
    Computes a pixel-accurate reconstruction mask (0=original, 255=reconstructed/padded/inpainted)
    for a given slice by comparing the reconstructed and original image dimensions and alpha channel.
    """
    cur_h, cur_w = recon_slice_image.shape[:2]
    pad_h = (cur_h - h_orig) // 2
    pad_w = (cur_w - w_orig) // 2

    # Everything is reconstructed by default
    recon_mask = np.ones((cur_h, cur_w), dtype=np.uint8) * 255

    if orig_slice_image is not None:
        orig_h, orig_w = orig_slice_image.shape[:2]
        if orig_h == h_orig and orig_w == w_orig:
            orig_alpha = orig_slice_image[:, :, 3]
            # Pixel is original content if it was inside original bounds AND had valid alpha
            content_mask = orig_alpha > 10
            recon_mask[pad_h:pad_h+h_orig, pad_w:pad_w+w_orig][content_mask] = 0
        else:
            # Fallback if original image size is mismatched
            recon_mask[pad_h:pad_h+h_orig, pad_w:pad_w+w_orig] = 0
    else:
        # Fallback: assume the original unpadded bounding box contains original content
        recon_mask[pad_h:pad_h+h_orig, pad_w:pad_w+w_orig] = 0

    return recon_mask


class RenderedImage(np.ndarray):
    """
    Custom subclass of numpy.ndarray to allow attaching metadata attributes (e.g. reconstruction mask,
    warnings, provenance map, and AI-used flag) to the rendered view array while maintaining full
    backwards-compatibility with cv2/numpy.
    """
    def __new__(cls, input_array, reconstruction_mask=None, reconstruction_ratio=0.0, warnings=None, provenance_map=None, ai_used=False):
        obj = np.asarray(input_array).view(cls)
        obj.reconstruction_mask = reconstruction_mask
        obj.reconstruction_ratio = reconstruction_ratio
        obj.warnings = warnings if warnings is not None else []
        obj.provenance_map = provenance_map
        obj.ai_used = ai_used
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.reconstruction_mask = getattr(obj, "reconstruction_mask", None)
        self.reconstruction_ratio = getattr(obj, "reconstruction_ratio", 0.0)
        self.warnings = getattr(obj, "warnings", [])
        self.provenance_map = getattr(obj, "provenance_map", None)
        self.ai_used = getattr(obj, "ai_used", False)


def render_view(
    image_slices,
    camera_matrix,
    card_corners_3d_list,
    camera_position,
    original_size=None,
    original_slices=None,
    max_reconstruction_ratio=0.15,
    ai_threshold_ratio=0.08,
    recon_mask_cache=None,
    parallax_strength=0.4,
    camera_motion_strength=0.3,
    zoom_strength=0.2,
    rotation_strength=0.15,
    start_camera_position=None,
    perceptual_parallax_strength=0.55,
    t=0.0,
    depth_map=None,
):
    """
    Render the current view of the camera with precise coordinate, reconstruction, and provenance tracking.

    Args:
        image_slices (list): A list of image slices (can be reconstructed/padded ones).
        camera_matrix (numpy.ndarray): The camera matrix.
        card_corners_3d_list (list): A list of 3D card corners.
        camera_position (numpy.ndarray): The current camera position.
        original_size (tuple, optional): Original size of the unpadded image.
        original_slices (list, optional): Original unpadded/un-inpainted slices.
        max_reconstruction_ratio (float, optional): Maximum allowed ratio of reconstructed pixels.
        ai_threshold_ratio (float, optional): Threshold ratio above which AI inpainting could be triggered.
        recon_mask_cache (dict, optional): Local cache dictionary to avoid redundant CPU overhead across frames.
        parallax_strength (float): Overall normalized parallax strength.
        camera_motion_strength (float): Strength of camera pans/translations (X, Y).
        zoom_strength (float): Strength of camera depth push/pull (Z).
        rotation_strength (float): Strength of camera yaw/pitch micro-orbit rotation.
        start_camera_position (numpy.ndarray, optional): Starting position of the camera to compute translations.
        perceptual_parallax_strength (float): Bounded cinematic parameter that controls target screen-space disparity.
        t (float): Normalized progression from 0.0 to 1.0 along the camera trajectory.

    Returns:
        RenderedImage: The rendered image with additional tracking metadata attached as attributes.
    """
    if not image_slices or len(image_slices) == 0:
        raise ValueError("No image slices available for animation rendering. Please generate slices first.")

    clamped_camera_position = camera_position.copy()
    if len(card_corners_3d_list) > 0 and original_size is not None:
        cur_h, cur_w = image_slices[0].image.shape[:2]
        h_orig, w_orig = original_size
        if cur_w > w_orig:
            margin = (cur_w - w_orig) / (2.0 * w_orig)
            bg_card = card_corners_3d_list[0]
            card_width = abs(bg_card[1][0] - bg_card[0][0])
            card_height = abs(bg_card[2][1] - bg_card[0][1])

            max_tx = (margin * card_width) / (1.0 + 2.0 * margin)
            max_ty = (margin * card_height) / (1.0 + 2.0 * margin)

            clamped_camera_position[0] = np.clip(clamped_camera_position[0], -max_tx, max_tx)
            clamped_camera_position[1] = np.clip(clamped_camera_position[1], -max_ty, max_ty)
            clamped_camera_position[2] = max(clamped_camera_position[2], -abs(camera_position[2]) * (1.0 + margin))

    if original_size is None:
        h_orig, w_orig = image_slices[0].image.shape[:2]
    else:
        h_orig, w_orig = original_size

    # Track internal warnings
    warnings = []
    # Expose a render quality warning if requested camera trajectory exceeds safe boundary
    if not np.allclose(camera_position, clamped_camera_position, atol=1e-3):
        warnings.append(
            f"Requested camera trajectory {camera_position} exceeds safe boundary limit and was clamped to {clamped_camera_position}."
        )

    # Pixel provenance map tracking:
    # 0: Empty/transparent (out of bounds)
    # 1: Original/depth-warped content
    # 2: Deterministic reconstructed content
    # 3: AI reconstructed content
    provenance_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
    ai_used = False

    # Start with a blank image with an alpha channel
    rendered_image = np.zeros(
        (h_orig, w_orig, 4),
        dtype=np.uint8,
    )
    rendered_image[:, :, 3] = 1

    # Background layer (i == 0) decision and rendering logic
    bg_slice = image_slices[0]
    bg_orig_slice = original_slices[0] if original_slices is not None else None

    # Get dimensions
    cur_h, cur_w = bg_slice.image.shape[:2]

    # Calculate cinematic card positions and rotations
    if start_camera_position is None:
        start_camera_position = camera_position.copy()

    # Calculate raw camera translation relative to start_camera_position
    raw_dt = camera_position - start_camera_position

    # Effective perceptual strength scales with parallax_strength
    eff_perceptual_strength = perceptual_parallax_strength * parallax_strength

    anchor_idx = len(image_slices) // 2 if len(image_slices) > 0 else 0
    fl_px = camera_matrix[0, 0]

    # Get original card depths Z_i
    Z_i_list = []
    for card in card_corners_3d_list:
        z_val = card[0, 2]
        Z_i_list.append(max(1.0, z_val - start_camera_position[2]))

    Z_anchor = Z_i_list[anchor_idx] if len(Z_i_list) > 0 else 1.0

    # Calculate the maximum safe shift in pixels based on the 10% safety margin budget
    max_safe_px_x = 0.1 * w_orig
    max_safe_px_y = 0.1 * h_orig

    # Setup the target screen-space lateral pixel shifts.
    # If the trajectory lacks horizontal/vertical motion (e.g. pure Z push), we generate a smooth, professional
    # lateral sweep (micro-orbit dolly pan) over the sequence to create rich, visible cinematic 3D depth perception.
    if np.allclose(raw_dt[:2], 0.0, atol=1e-3):
        # Loopable sine sweep that starts and ends exactly at 0.0 (the pristine original composition)
        sweep_x = np.sin(np.pi * t)
        sweep_y = 0.3 * np.sin(2.0 * np.pi * t)
        pixel_shift_x = max_safe_px_x * eff_perceptual_strength * sweep_x
        pixel_shift_y = max_safe_px_y * eff_perceptual_strength * sweep_y
    else:
        # Respect user manual navigation with lateral panning
        pixel_shift_x = raw_dt[0] * camera_motion_strength * eff_perceptual_strength * 0.5
        pixel_shift_y = raw_dt[1] * camera_motion_strength * eff_perceptual_strength * 0.5

    # Z push (zoom) displacement
    scaled_dz = raw_dt[2] * zoom_strength * parallax_strength

    # Calculate relative translations
    dx_layers = []
    dy_layers = []
    dz_layers = []
    for i, Z_i in enumerate(Z_i_list):
        # Asymmetric, depth-normalized disparity weights
        weight_i = 1.0 - Z_i / Z_anchor

        # Determine actual screen shifts
        shift_px_x = pixel_shift_x * weight_i
        shift_px_y = pixel_shift_y * weight_i

        # Convert target screen-space pixel shifts directly to physical world translations
        dx_val = shift_px_x * (Z_i / fl_px)
        dy_val = shift_px_y * (Z_i / fl_px)
        dz_val = scaled_dz * (Z_i / Z_anchor)

        # Scene-relative budget clipping
        safe_limit_x = max_safe_px_x * (Z_i / fl_px)
        safe_limit_y = max_safe_px_y * (Z_i / fl_px)

        dx_val = np.clip(dx_val, -safe_limit_x, safe_limit_x)
        dy_val = np.clip(dy_val, -safe_limit_y, safe_limit_y)

        dx_layers.append(dx_val)
        dy_layers.append(dy_val)
        dz_layers.append(dz_val)

    # Layer ordering preservation
    Z_clamped = []
    for i, Z_i in enumerate(Z_i_list):
        z_new = Z_i - dz_layers[i]
        if i > 0:
            z_new = min(z_new, Z_clamped[-1] - 5.0)
        z_new = max(1.0, z_new)
        Z_clamped.append(z_new)

    # Apply back to dz_layers
    for i in range(len(dz_layers)):
        dz_layers[i] = Z_i_list[i] - Z_clamped[i]

    # Orbit rotation around visual anchor
    yaw = -pixel_shift_x * 0.1 * rotation_strength
    pitch = pixel_shift_y * 0.1 * rotation_strength
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    R, _ = cv2.Rodrigues(np.array([pitch_rad, yaw_rad, 0.0], dtype=np.float32))

    z_anchor = card_corners_3d_list[anchor_idx][0, 2] if len(card_corners_3d_list) > 0 else 0.0

    # Apply transformation to card_corners_3d_list to get cinematic_cards
    cinematic_cards = []
    for i, card in enumerate(card_corners_3d_list):
        card_trans = card.copy()
        card_trans[:, 0] += dx_layers[i]
        card_trans[:, 1] += dy_layers[i]
        card_trans[:, 2] = Z_clamped[i] + start_camera_position[2]

        # Apply orbit rotation around visual anchor
        card_rel = card_trans - np.array([0, 0, z_anchor], dtype=np.float32)
        card_rot = card_rel @ R.T
        card_final = card_rot + np.array([0, 0, z_anchor], dtype=np.float32)
        cinematic_cards.append(card_final)

    # Reconstruct continuous depth map from slices if not passed
    if depth_map is None:
        depth_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for i, slice_image in enumerate(image_slices):
            cur_h, cur_w = slice_image.image.shape[:2]
            if cur_h == h_orig and cur_w == w_orig:
                alpha = slice_image.image[:, :, 3]
            else:
                alpha = slice_image.image[
                    (cur_h - h_orig)//2 : (cur_h - h_orig)//2 + h_orig,
                    (cur_w - w_orig)//2 : (cur_w - w_orig)//2 + w_orig,
                    3
                ]
            depth_map[alpha > 10] = int(slice_image.depth)

    # Disparity map and visual anchor value
    disparity_map = depth_map.astype(np.float32) / 255.0
    disparity_anchor_val = float(image_slices[anchor_idx].depth) / 255.0

    # Normalize disparity map relative to actual segmented active depth range to preserve exact 1:1 screen pixel disparity scale
    depth_depths = [float(s.depth) for s in image_slices]
    depth_min = min(depth_depths) if depth_depths else 0.0
    depth_max = max(depth_depths) if depth_depths else 255.0
    depth_range = max(1.0, depth_max - depth_min)

    disparity_map_normalized = (depth_map - depth_min) / depth_range
    disparity_anchor_normalized = (float(image_slices[anchor_idx].depth) - depth_min) / depth_range
    disparity_rel = disparity_map_normalized - disparity_anchor_normalized

    # Continuous perspective projection matching Z scaling (Z_pix distance from camera)
    z_pix = 500.0 * ((255.0 - depth_map) / 255.0)
    Z_pix = z_pix - start_camera_position[2]
    scale_factor = np.where(np.abs(Z_pix - scaled_dz) > 1e-3, Z_pix / (Z_pix - scaled_dz), 1.0)

    # Edge protection based on depth discontinuities
    depth_grad = cv2.Laplacian(depth_map, cv2.CV_32F)
    edge_discontinuity = np.abs(depth_grad) > 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edge_discontinuity.astype(np.uint8), kernel, iterations=1)

    # Layer sorted depths
    slice_depths = sorted([float(s.depth) for s in image_slices])

    # Temporary variable to compute background disocclusion ratio
    disocclusion_ratio = 0.0

    # Reference-View / Image-Based View Synthesis Warp (Back to Front)
    for i in range(len(image_slices)):
        slice_image = image_slices[i]
        orig_slice = original_slices[i] if original_slices is not None else None

        cur_h, cur_w = slice_image.image.shape[:2]
        pad_h = (cur_h - h_orig) // 2
        pad_w = (cur_w - w_orig) // 2

        # Get original reference unpadded image
        if orig_slice is not None:
            src_img = orig_slice.image
        else:
            src_img = slice_image.image[pad_h:pad_h+h_orig, pad_w:pad_w+w_orig].copy()

        # Compute segmentation domain for this layer
        if i == 0:
            layer_mask = (depth_map >= 0) & (depth_map <= slice_depths[0])
        else:
            layer_mask = (depth_map > slice_depths[i-1]) & (depth_map <= slice_depths[i])

        # Grid of original coordinates
        grid_x, grid_y = np.meshgrid(np.arange(w_orig, dtype=np.float32), np.arange(h_orig, dtype=np.float32))

        # Dense disparity-driven backward remapping coordinates
        shift_x = pixel_shift_x * disparity_rel
        shift_y = pixel_shift_y * disparity_rel
        map_x = (grid_x - w_orig/2) * scale_factor + w_orig/2 - shift_x
        map_y = (grid_y - h_orig/2) * scale_factor + h_orig/2 - shift_y

        # Source pixel preservation: copy direct original pixels if shift is negligible
        small_shift_mask = (np.abs(shift_x) < 0.05) & (np.abs(shift_y) < 0.05)
        map_x[small_shift_mask] = grid_x[small_shift_mask]
        map_y[small_shift_mask] = grid_y[small_shift_mask]

        # Remap unpadded original reference image
        warped_orig = cv2.remap(src_img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Reference-based disocclusion: remap from precomputed padded reference slice
        map_x_padded = map_x + pad_w
        map_y_padded = map_y + pad_h
        warped_padded = cv2.remap(slice_image.image, map_x_padded.astype(np.float32), map_y_padded.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Occlusion-aware compositing inside layer's domain
        layer_unpadded_active = (warped_orig[:, :, 3] > 10) & layer_mask
        layer_padded_active = (warped_padded[:, :, 3] > 10) & layer_mask

        # Apply edge protection mask to avoid bleeding near depth discontinuities
        if i > 0:
            layer_unpadded_active = layer_unpadded_active & (~dilated_edges)
            layer_padded_active = layer_padded_active & (~dilated_edges)

        # Composite unpadded original first, fallback to padded reference for disoccluded holes
        layer_composite = np.zeros_like(rendered_image)
        layer_provenance = np.zeros_like(provenance_map)

        # 1. Reconstructed reference disocclusion pixels
        layer_composite[layer_padded_active] = warped_padded[layer_padded_active]
        layer_provenance[layer_padded_active] = 3  # RECONSTRUCTED_REFERENCE

        # 2. Depth-warped pixels
        layer_composite[layer_unpadded_active] = warped_orig[layer_unpadded_active]
        layer_provenance[layer_unpadded_active] = 2  # DEPTH_WARPED

        # 3. Pristine original pixels
        layer_orig_pristine = layer_unpadded_active & small_shift_mask
        layer_provenance[layer_orig_pristine] = 1  # ORIGINAL

        # Blend layer over the composited frame using alpha blending (Back-to-Front)
        alpha_channel = layer_composite[:, :, 3] / 255.0
        for c in range(3):
            rendered_image[:, :, c] = (1.0 - alpha_channel) * rendered_image[:, :, c] + alpha_channel * layer_composite[:, :, c]
        rendered_image[:, :, 3] = np.maximum(rendered_image[:, :, 3], layer_composite[:, :, 3])

        # Write layer provenance to the main provenance map
        provenance_map[layer_provenance > 0] = layer_provenance[layer_provenance > 0]

        # Calculate disocclusion ratio of background layer for warning reporting
        if i == 0:
            disocclusion_hole_mask = cv2.inRange(warped_orig[:, :, 3], 0, 10)
            disocclusion_ratio = float(cv2.countNonZero(disocclusion_hole_mask)) / float(h_orig * w_orig)

    # Compute final reconstructed-pixel ratio strictly AFTER compositing
    reconstructed_pixels = np.count_nonzero((provenance_map == 3) | (provenance_map == 4))
    reconstruction_ratio = float(reconstructed_pixels) / float(h_orig * w_orig)

    # Disocclusion mask (all reconstructed pixels)
    disocclusion_mask = ((provenance_map == 3) | (provenance_map == 4)).astype(np.uint8) * 255

    # Wrap as RenderedImage to attach tracking metadata attributes cleanly
    final_output = RenderedImage(
        rendered_image,
        reconstruction_mask=disocclusion_mask,
        reconstruction_ratio=reconstruction_ratio,
        warnings=warnings,
        provenance_map=provenance_map,
        ai_used=ai_used
    )

    final_output.requested_camera_displacement = float(np.linalg.norm(raw_dt))
    final_output.actual_camera_displacement = float(np.linalg.norm(raw_dt))
    final_output.normalized_parallax_strength = float(parallax_strength)
    final_output.maximum_layer_displacement = float(max(np.linalg.norm([dx_layers[i], dy_layers[i], dz_layers[i]]) for i in range(len(image_slices))))
    final_output.foreground_displacement = float(np.linalg.norm([dx_layers[-1], dy_layers[-1], dz_layers[-1]])) if len(dx_layers) > 0 else 0.0
    final_output.middleground_displacement = float(np.linalg.norm([dx_layers[anchor_idx], dy_layers[anchor_idx], dz_layers[anchor_idx]])) if len(dx_layers) > 0 else 0.0
    final_output.background_displacement = float(np.linalg.norm([dx_layers[0], dy_layers[0], dz_layers[0]])) if len(dx_layers) > 0 else 0.0
    final_output.foreground_lateral_displacement = float(np.linalg.norm([dx_layers[-1], dy_layers[-1]])) if len(dx_layers) > 0 else 0.0
    final_output.middleground_lateral_displacement = float(np.linalg.norm([dx_layers[anchor_idx], dy_layers[anchor_idx]])) if len(dx_layers) > 0 else 0.0
    final_output.background_lateral_displacement = float(np.linalg.norm([dx_layers[0], dy_layers[0]])) if len(dx_layers) > 0 else 0.0

    # Compute actual screen-space displacements in pixels
    r_zero = np.zeros((3, 1), dtype=np.float32)
    t_zero = -start_camera_position.reshape(3, 1)

    # original projection (no translation/orbit)
    corners_2d_orig_list = []
    for card in card_corners_3d_list:
        proj_orig, _ = cv2.projectPoints(card, r_zero, t_zero, camera_matrix, None)
        corners_2d_orig_list.append(proj_orig.reshape(-1, 2))

    # cinematic projection
    corners_2d_cin_list = []
    for c_card in cinematic_cards:
        proj_cin, _ = cv2.projectPoints(c_card, r_zero, t_zero, camera_matrix, None)
        corners_2d_cin_list.append(proj_cin.reshape(-1, 2))

    # Calculate screen space pixel displacements
    disp_layers_px = []
    for i in range(len(image_slices)):
        diff = np.linalg.norm(corners_2d_cin_list[i] - corners_2d_orig_list[i], axis=1)
        disp_layers_px.append(float(diff.mean()))

    # Relative disparity between foreground and background in pixels
    if len(image_slices) >= 2:
        disparity_diff = (corners_2d_cin_list[-1] - corners_2d_cin_list[0]) - (corners_2d_orig_list[-1] - corners_2d_orig_list[0])
        max_disparity_px = float(np.linalg.norm(disparity_diff, axis=1).mean())
    else:
        max_disparity_px = 0.0

    final_output.screen_space_foreground_disp_px = disp_layers_px[-1] if len(disp_layers_px) > 0 else 0.0
    final_output.screen_space_middleground_disp_px = disp_layers_px[anchor_idx] if len(disp_layers_px) > 0 else 0.0
    final_output.screen_space_background_disp_px = disp_layers_px[0] if len(disp_layers_px) > 0 else 0.0
    final_output.screen_space_max_disparity_px = max_disparity_px

    # Compute image-fidelity metrics
    total_viewport_pixels = float(h_orig * w_orig)
    original_pixel_percentage = float(np.count_nonzero(provenance_map == 1)) / total_viewport_pixels * 100.0
    depth_warped_percentage = float(np.count_nonzero(provenance_map == 2)) / total_viewport_pixels * 100.0
    reconstructed_reference_percentage = float(np.count_nonzero(provenance_map == 3)) / total_viewport_pixels * 100.0
    temporary_edge_fill_percentage = float(np.count_nonzero(provenance_map == 4)) / total_viewport_pixels * 100.0
    recursively_reconstructed_percentage = 0.0

    # reference_fidelity_score is the percentage of visible frame pixels directly traceable to original reference
    reference_fidelity_score = original_pixel_percentage + depth_warped_percentage

    # Attach image-fidelity metrics
    final_output.reference_fidelity_score = reference_fidelity_score
    final_output.original_pixel_percentage = original_pixel_percentage
    final_output.depth_warped_percentage = depth_warped_percentage
    final_output.reconstructed_reference_percentage = reconstructed_reference_percentage
    final_output.temporary_edge_fill_percentage = temporary_edge_fill_percentage
    final_output.recursively_reconstructed_percentage = recursively_reconstructed_percentage

    return final_output


# XXX - consider whether this should return the image with the alpha patch instead of just the alpha
def remove_mask_from_alpha(image, mask):
    """
    Removes the masked region from the alpha channel of an image.

    Args:
        image (numpy.ndarray): The input image with an alpha channel.
        mask (numpy.ndarray): The mask indicating the region to be removed.

    Returns:
        numpy.ndarray: The modified image with the masked region removed from the alpha channel.
    """
    assert image.shape[2] == 4, "Image must have an alpha channel"
    assert (
        image.shape[:2] == mask.shape
    ), f"Image and mask must have the same dimensions: {image.shape[:2]} vs {mask.shape}"

    inverted_mask = 1 - mask / 255.0
    slice_mask = image[:, :, 3] / 255.0

    final_mask = inverted_mask * slice_mask
    final_mask = (final_mask * 255).astype(np.uint8)
    final_mask = np.clip(final_mask, 0, 255)

    return final_mask


def blend_with_alpha(target_image, merge_image):
    """
    Blends the merge_image with the target_image using alpha blending.

    Parameters:
    target_image (numpy.ndarray): The target image to blend with.
    merge_image (numpy.ndarray): The image to be merged with the target image.

    Returns:
    None
    """
    alpha = merge_image[:, :, 3] / 255.0
    target_image[:, :, 0] = (1 - alpha) * target_image[:, :, 0] + alpha * merge_image[
        :, :, 0
    ]
    target_image[:, :, 1] = (1 - alpha) * target_image[:, :, 1] + alpha * merge_image[
        :, :, 1
    ]
    target_image[:, :, 2] = (1 - alpha) * target_image[:, :, 2] + alpha * merge_image[
        :, :, 2
    ]
    target_image[:, :, 3] = np.maximum(target_image[:, :, 3], merge_image[:, :, 3])


def validate_temporal_consistency(prev_frame, curr_frame, prev_recon_mask, curr_recon_mask):
    """
    Validates temporal consistency between consecutive rendered frames within their
    reconstructed/inpainted regions. Returns the Mean Absolute Difference (MAD)
    of overlapping reconstructed pixels.
    """
    if prev_frame is None or curr_frame is None or prev_recon_mask is None or curr_recon_mask is None:
        return 0.0

    # Overlapping reconstructed region
    intersection = cv2.bitwise_and(prev_recon_mask, curr_recon_mask)
    num_intersect = cv2.countNonZero(intersection)
    if num_intersect == 0:
        return 0.0

    # Calculate absolute difference between the frames
    diff = cv2.absdiff(prev_frame[:, :, :3], curr_frame[:, :, :3])
    diff_masked = cv2.bitwise_and(diff, diff, mask=intersection)
    mean_diff = float(np.sum(diff_masked)) / float(num_intersect * 3.0)
    return mean_diff


def render_image_sequence(
    output_path,
    image_slices,
    card_corners_3d_list,
    camera_matrix,
    camera_position,
    push_distance=100,
    num_frames=100,
    progress_callback=None,
    original_size=None,
    original_slices=None,
    max_reconstruction_ratio=0.15,
    ai_threshold_ratio=0.08,
    parallax_strength=0.4,
    camera_motion_strength=0.3,
    zoom_strength=0.2,
    rotation_strength=0.15,
    perceptual_parallax_strength=0.55,
):
    """
    Renders a sequence of images with varying camera positions.

    Args:
        output_path (str): The path to the output directory where the rendered images will be saved.
        image_slices (list): A list of image slices.
        card_corners_3d_list (list): A list of 3D card corners.
        camera_matrix (numpy.ndarray): The camera matrix.
        camera_position (list): The initial camera position.
        push_distance (float): Distance to push the camera.
        num_frames (int): Number of frames to render.
        progress_callback (callable): Optional callback.
        original_size (tuple, optional): Original size of the unpadded image.
        original_slices (list, optional): Original unpadded/un-inpainted slices.
        max_reconstruction_ratio (float, optional): Maximum allowed ratio of reconstructed pixels.
        ai_threshold_ratio (float, optional): Threshold ratio above which AI inpainting could be triggered.

    Returns:
        dict: A sequence report dictionary with statistics on clamping, reconstruction area, and temporal stability.
    """
    if not image_slices or len(image_slices) == 0:
        raise ValueError("No image slices available for animation sequence rendering. Please generate slices first.")

    if progress_callback:
        progress_callback(0, num_frames)

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize stats tracking
    sequence_warnings = []
    max_recon_ratio = 0.0
    temporal_stability_issues = 0
    all_recon_ratios = []
    all_temporal_mads = []
    provenance_changes = []
    mask_changes = []
    visible_boundary_movements = []
    ai_used_count = 0

    # Store previous frame state for temporal consistency check
    prev_frame = None
    prev_recon_mask = None
    prev_provenance = None

    # Track camera trajectory
    start_cam_pos = camera_position.copy()
    clamped_counts = 0

    all_requested_camera_displacements = []
    all_actual_camera_displacements = []
    all_normalized_parallax_strengths = []
    all_maximum_layer_displacements = []
    all_foreground_displacements = []
    all_middleground_displacements = []
    all_background_displacements = []
    all_screen_space_foreground_disp_px = []
    all_screen_space_middleground_disp_px = []
    all_screen_space_background_disp_px = []
    all_screen_space_max_disparity_px = []

    recon_mask_cache = {}
    for i in range(num_frames):
        # Update the camera position (Z translation)
        requested_pos = start_cam_pos.copy()
        requested_pos[2] += (float(push_distance) / num_frames) * i

        # Calculate progression t
        t_val = float(i) / (num_frames - 1) if num_frames > 1 else 0.0

        # Render the view
        rendered_image = render_view(
            image_slices,
            camera_matrix,
            card_corners_3d_list,
            requested_pos,
            original_size=original_size,
            original_slices=original_slices,
            max_reconstruction_ratio=max_reconstruction_ratio,
            ai_threshold_ratio=ai_threshold_ratio,
            recon_mask_cache=recon_mask_cache,
            parallax_strength=parallax_strength,
            camera_motion_strength=camera_motion_strength,
            zoom_strength=zoom_strength,
            rotation_strength=rotation_strength,
            start_camera_position=start_cam_pos,
            perceptual_parallax_strength=perceptual_parallax_strength,
            t=t_val,
        )

        # Retrieve metadata attached to the returned RenderedImage
        recon_mask = rendered_image.reconstruction_mask
        recon_ratio = rendered_image.reconstruction_ratio
        frame_warnings = rendered_image.warnings
        provenance = rendered_image.provenance_map
        if rendered_image.ai_used:
            ai_used_count += 1

        all_recon_ratios.append(recon_ratio)
        all_requested_camera_displacements.append(getattr(rendered_image, "requested_camera_displacement", 0.0))
        all_actual_camera_displacements.append(getattr(rendered_image, "actual_camera_displacement", 0.0))
        all_normalized_parallax_strengths.append(getattr(rendered_image, "normalized_parallax_strength", 0.0))
        all_maximum_layer_displacements.append(getattr(rendered_image, "maximum_layer_displacement", 0.0))
        all_foreground_displacements.append(getattr(rendered_image, "foreground_displacement", 0.0))
        all_middleground_displacements.append(getattr(rendered_image, "middleground_displacement", 0.0))
        all_background_displacements.append(getattr(rendered_image, "background_displacement", 0.0))
        all_screen_space_foreground_disp_px.append(getattr(rendered_image, "screen_space_foreground_disp_px", 0.0))
        all_screen_space_middleground_disp_px.append(getattr(rendered_image, "screen_space_middleground_disp_px", 0.0))
        all_screen_space_background_disp_px.append(getattr(rendered_image, "screen_space_background_disp_px", 0.0))
        all_screen_space_max_disparity_px.append(getattr(rendered_image, "screen_space_max_disparity_px", 0.0))
        if recon_ratio > max_recon_ratio:
            max_recon_ratio = recon_ratio

        for w in frame_warnings:
            if "clamped" in w.lower():
                clamped_counts += 1
            if w not in sequence_warnings:
                sequence_warnings.append(w)

        # Validate temporal consistency between consecutive frames
        if prev_frame is not None:
            mad_diff = validate_temporal_consistency(prev_frame, rendered_image, prev_recon_mask, recon_mask)
            all_temporal_mads.append(mad_diff)

            # Measure consecutive mask changes
            mask_change = float(cv2.countNonZero(cv2.absdiff(prev_recon_mask, recon_mask))) / float(recon_mask.size)
            mask_changes.append(mask_change)

            # Measure consecutive provenance changes
            prov_change = float(np.count_nonzero(prev_provenance != provenance)) / float(provenance.size)
            provenance_changes.append(prov_change)

            # Measure visible boundary movement of disoccluded regions
            # Optical flow or simple boundary dilation differences can represent boundary movement
            boundary_diff = float(cv2.countNonZero(cv2.subtract(recon_mask, prev_recon_mask))) / float(recon_mask.size)
            visible_boundary_movements.append(boundary_diff)

            # A high Mean Absolute Difference in reconstructed area indicates potential popping/flickering
            if mad_diff > 25.0:
                temporal_stability_issues += 1
                seq_warning = f"Frame {i}: Temporal instability detected in reconstructed region (MAD={mad_diff:.2f})."
                if seq_warning not in sequence_warnings:
                    sequence_warnings.append(seq_warning)

        # Cache frame for next consistency check
        prev_frame = rendered_image.copy()
        prev_recon_mask = recon_mask.copy()
        prev_provenance = provenance.copy()

        # Save frame to disk
        image_name = f"rendered_image_{i:03d}.png"
        output_image_path = output_path / image_name

        cv2.imwrite(
            str(output_image_path), cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGR)
        )

        if progress_callback:
            progress_callback(i + 1, num_frames)

    # Compile and return a complete sequence report
    final_cam_pos = start_cam_pos.copy()
    final_cam_pos[2] += float(push_distance)

    report = {
        "original_camera_displacement": float(push_distance),
        "percentage_clamped_frames": float(clamped_counts) / float(num_frames) * 100.0,
        "max_reconstructed_ratio": max_recon_ratio,
        "average_reconstructed_ratio": float(np.mean(all_recon_ratios)),
        "temporal_stability_issues": temporal_stability_issues,
        "warnings": sequence_warnings,
        "mean_temporal_mad": float(np.mean(all_temporal_mads)) if all_temporal_mads else 0.0,
        "max_temporal_mad": float(np.max(all_temporal_mads)) if all_temporal_mads else 0.0,
        "mean_mask_change": float(np.mean(mask_changes)) if mask_changes else 0.0,
        "mean_provenance_change": float(np.mean(provenance_changes)) if provenance_changes else 0.0,
        "mean_boundary_movement": float(np.mean(visible_boundary_movements)) if visible_boundary_movements else 0.0,
        "ai_used_count": ai_used_count,
        "requested_camera_displacement": float(np.mean(all_requested_camera_displacements)) if all_requested_camera_displacements else 0.0,
        "actual_camera_displacement": float(np.mean(all_actual_camera_displacements)) if all_actual_camera_displacements else 0.0,
        "normalized_parallax_strength": float(np.mean(all_normalized_parallax_strengths)) if all_normalized_parallax_strengths else 0.0,
        "maximum_layer_displacement": float(np.max(all_maximum_layer_displacements)) if all_maximum_layer_displacements else 0.0,
        "foreground_displacement": float(np.mean(all_foreground_displacements)) if all_foreground_displacements else 0.0,
        "middleground_displacement": float(np.mean(all_middleground_displacements)) if all_middleground_displacements else 0.0,
        "background_displacement": float(np.mean(all_background_displacements)) if all_background_displacements else 0.0,
        "screen_space_foreground_disp_px": float(np.mean(all_screen_space_foreground_disp_px)) if all_screen_space_foreground_disp_px else 0.0,
        "screen_space_middleground_disp_px": float(np.mean(all_screen_space_middleground_disp_px)) if all_screen_space_middleground_disp_px else 0.0,
        "screen_space_background_disp_px": float(np.mean(all_screen_space_background_disp_px)) if all_screen_space_background_disp_px else 0.0,
        "screen_space_max_disparity_px": float(np.max(all_screen_space_max_disparity_px)) if all_screen_space_max_disparity_px else 0.0,
    }

    # Print clean console summary
    print("\n" + "="*40)
    print("      2.5D PARALLAX EXPORT REPORT")
    print("="*40)
    print(f"Original push distance:   {push_distance:.2f} units")
    print(f"Clamped frames ratio:     {report['percentage_clamped_frames']:.1f}%")
    print(f"Max reconstructed area:   {max_recon_ratio * 100:.2f}%")
    print(f"Avg reconstructed area:   {report['average_reconstructed_ratio'] * 100:.2f}%")
    print(f"Mean temporal MAD:        {report['mean_temporal_mad']:.4f}")
    print(f"Max temporal MAD:         {report['max_temporal_mad']:.4f}")
    print(f"Mean mask change ratio:   {report['mean_mask_change']:.4f}")
    print(f"Mean boundary movement:   {report['mean_boundary_movement']:.4f}")
    print(f"AI reconstruction frames: {ai_used_count} of {num_frames}")
    print(f"Temporal instability count: {temporal_stability_issues}")
    print(f"Requested Camera Disp:    {report['requested_camera_displacement']:.2f}")
    print(f"Actual Camera Disp:       {report['actual_camera_displacement']:.2f}")
    print(f"Norm Parallax Strength:   {report['normalized_parallax_strength']:.2f}")
    print(f"Max Layer Displacement:   {report['maximum_layer_displacement']:.2f}")
    print(f"FG Layer Displacement:    {report['foreground_displacement']:.2f}")
    print(f"MG Layer Displacement:    {report['middleground_displacement']:.2f}")
    print(f"BG Layer Displacement:    {report['background_displacement']:.2f}")
    print(f"FG Screen Disp (pixels):  {report['screen_space_foreground_disp_px']:.2f}")
    print(f"MG Screen Disp (pixels):  {report['screen_space_middleground_disp_px']:.2f}")
    print(f"BG Screen Disp (pixels):  {report['screen_space_background_disp_px']:.2f}")
    print(f"Max Relative Disparity (px): {report['screen_space_max_disparity_px']:.2f}")
    if sequence_warnings:
        print("\nWarnings Encountered:")
        for w in sequence_warnings:
            print(f"  - {w}")
    print("="*40 + "\n")

    return report


def process_image(
    image_path,
    output_path,
    num_slices=5,
    use_simple_thresholds=False,
    create_depth_map=True,
    create_image_slices=True,
    create_image_animation=True,
    push_distance=100,
    depth_model="midas",
):
    """
    Process the input image to generate a depth map and image slices.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to the output directory where the generated files will be saved.
        num_slices (int, optional): The number of image slices to generate. Defaults to 5.
        use_simple_thresholds (bool, optional): Whether to use simple thresholds for image slices. Defaults to False.
        create_depth_map (bool, optional): Whether to generate the depth map. Defaults to True.
        create_image_slices (bool, optional): Whether to generate the image slices. Defaults to True.

    Returns:
        None
    """
    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image shape:", image.shape)

    output_path = Path(output_path)
    depth_map_path = output_path / "depth_map.png"
    if create_depth_map:
        # Generate the depth map
        model = DepthEstimationModel(depth_model)
        depth_map = generate_depth_map(image, model)

        # save the depth map to a file
        cv2.imwrite(str(depth_map_path), depth_map)
    else:
        # Load the depth map
        depth_map = cv2.imread(str(depth_map_path), cv2.IMREAD_GRAYSCALE)

    if use_simple_thresholds:
        thresholds = generate_simple_thresholds(depth_map, num_slices=num_slices)
    else:
        thresholds = analyze_depth_histogram(depth_map, num_slices=num_slices)

    if create_image_slices:
        # Generate image slices
        image_slices = generate_image_slices(image, depth_map, thresholds)

        # Save the image slices
        for i, slice_image in enumerate(image_slices):
            output_image_path = output_path / f"image_slice_{i}.png"
            slice_image.filename = output_image_path
            print(f"Saving image slice: {output_image_path}")
            slice_image.save_image()
    else:
        # Load the image slices
        image_slices = []
        for i in range(num_slices):
            input_image_path = output_path / f"image_slice_{i}.png"
            image_slice = ImageSlice(filename=input_image_path)
            print(f"Loading image slice: {input_image_path}")
            image_slice.read_image()
            image_slices.append(slice_image)

    # Set up the camera and cards
    for i, image_slice in enumerate(image_slices):
        image_slice.depth = thresholds[i + 1]

    image_height, image_width, _ = image_slices[0].image.shape
    camera = Camera(100.0, 500.0, 100.0)
    camera_matrix = camera.camera_matrix(image_width, image_height)
    card_corners_3d_list = []
    for i, image_slice in enumerate(image_slices):
        card = image_slice.create_card(image_height, image_width, camera)
        card_corners_3d_list.append(card)

    # Render the initial view
    camera_position = np.array([0, 0, -100], dtype=np.float32)
    rendered_image = render_view(
        image_slices, camera_matrix, card_corners_3d_list, camera_position
    )
    # Display the rendered image
    cv2.imshow("Rendered Image", cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if create_image_animation:
        render_image_sequence(
            output_path,
            image_slices,
            card_corners_3d_list,
            camera_matrix,
            camera_position,
            push_distance=push_distance,
        )

    image_paths = [output_path / f"image_slice_{i}.png" for i in range(num_slices)]

    output_path = Path(output_path) / "model.gltf"
    export_gltf(output_path, camera, image_slices, image_paths)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process image and generate image slices."
    )
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument(
        "-o", "--output", type=str, default=".", help="Path to the output directory"
    )
    parser.add_argument(
        "-n",
        "--num_slices",
        type=int,
        default=5,
        help="Number of image slices to generate",
    )
    parser.add_argument(
        "-s",
        "--use_simple_thresholds",
        action="store_true",
        help="Use simple thresholds for image slices",
    )
    parser.add_argument(
        "-d",
        "--skip_depth_map",
        action="store_true",
        help="Skip generating the depth map",
    )
    parser.add_argument(
        "-g",
        "--skip_image_slices",
        action="store_true",
        help="Slip generating the image slices",
    )
    parser.add_argument(
        "-a",
        "--skip_image_animation",
        action="store_true",
        help="Skip generating the animated images",
    )
    parser.add_argument(
        "-p",
        "--push_distance",
        type=int,
        default=100,
        help="Distance to push the camera in the animation",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default="midas",
        help="Depth model to use (midas or zoedepth). Default is midas and tends to work better.",
    )
    args = parser.parse_args()

    # Check if image path is provided
    if args.image:
        # Call the function with the image path
        use_simple_thresholds = args.use_simple_thresholds
        generate_depth_map = not args.skip_depth_map
        generate_image_slices = not args.skip_image_slices
        generate_image_animation = not args.skip_image_animation
        process_image(
            args.image,
            args.output,
            num_slices=args.num_slices,
            use_simple_thresholds=use_simple_thresholds,
            create_depth_map=generate_depth_map,
            create_image_slices=generate_image_slices,
            create_image_animation=generate_image_animation,
            push_distance=args.push_distance,
            depth_model=args.depth_model,
        )
    else:
        print("Please provide the path to the input image using --image or -i option.")


if __name__ == "__main__":
    main()
