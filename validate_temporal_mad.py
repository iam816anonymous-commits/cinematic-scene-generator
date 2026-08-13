import cv2
import numpy as np
from pathlib import Path

def validate_disk_frames(frame_dir):
    frame_dir = Path(frame_dir)
    print(f"Loading frames from {frame_dir} for independent verification...")

    png_files = sorted(list(frame_dir.glob("rendered_image_*.png")))
    if len(png_files) < 2:
        print("Error: insufficient frames found for comparison!")
        return None

    all_mads = []
    max_mad = 0.0
    total_changed_pixels = 0
    total_pixels = 0

    for i in range(len(png_files) - 1):
        img_a = cv2.imread(str(png_files[i]))
        img_b = cv2.imread(str(png_files[i+1]))

        if img_a.shape != img_b.shape:
            print(f"Warning: dimension mismatch between frame {i} and {i+1}!")
            continue

        # Absolute RGB difference
        diff = cv2.absdiff(img_a, img_b)
        mean_diff = float(diff.mean())
        all_mads.append(mean_diff)

        if mean_diff > max_mad:
            max_mad = mean_diff

        # Count changed pixels (threshold difference > 0)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = cv2.countNonZero(gray_diff)
        total_changed_pixels += changed
        total_pixels += gray_diff.size

    mean_mad = float(np.mean(all_mads)) if all_mads else 0.0
    pct_changed = (float(total_changed_pixels) / float(total_pixels) * 100.0) if total_pixels > 0 else 0.0

    print("\n" + "="*50)
    print("      INDEPENDENT DISK-BASED RGB MAD REPORT")
    print("="*50)
    print(f"Total Frames Compared:     {len(png_files)}")
    print(f"Independent Mean MAD:      {mean_mad:.4f}")
    print(f"Independent Max MAD:       {max_mad:.4f}")
    print(f"Average Changed Pixels:    {pct_changed:.2f}%")
    print("="*50 + "\n")

    return {
        "mean_mad": mean_mad,
        "max_mad": max_mad,
        "pct_changed": pct_changed
    }

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rendered_output_300_0.7"
    validate_disk_frames(directory)
