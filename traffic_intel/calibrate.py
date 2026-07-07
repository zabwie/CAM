"""
Interactive camera calibrator.

Steps (each opens its own matplotlib window):
  1. Homography — click 4 road points that form a rectangle
  2. ROI polygon — click points around the drivable area (right-click to close)
  3. Speed trap — click 2 points for line A, 2 for line B; enter distance

Usage:
    python calibrate.py --video traffic.mp4 --output calib.json
    python calibrate.py --image frame.jpg --output calib.json
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(src: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"ERROR: cannot read {src}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _show_image(img, title) -> tuple:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Step 1: Homography
# ---------------------------------------------------------------------------

def _click_homography(img_rgb):
    """Click 4 points TL/TR/BR/BL. Return list of (u,v) or None."""
    pts = []

    def onclick(event):
        if event.inaxes is None or len(pts) >= 4:
            return
        pts.append((int(event.xdata), int(event.ydata)))
        ax.plot(event.xdata, event.ydata, "ro", markersize=6)
        ax.text(event.xdata + 5, event.ydata - 5, str(len(pts)),
                fontsize=12, color="red", fontweight="bold")
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "r-", linewidth=1)
        plt.draw()

    fig, ax = _show_image(img_rgb,
        "STEP 1: Click 4 road points forming a rectangle\n"
        "Order: 1=Top-Left  2=Top-Right  3=Bottom-Right  4=Bottom-Left\n"
        "Close window when done.")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(pts) < 4:
        print("Need exactly 4 points for homography. Skipping.")
        return None, 0, 0

    print(f"\nHomography points (image coords): {pts}")
    try:
        w = float(input("  Road WIDTH  (metres, perpendicular to traffic): ") or "3.7")
        h = float(input("  Road LENGTH (metres, along traffic direction): ") or "10.0")
    except (ValueError, EOFError):
        w, h = 3.7, 10.0
        print(f"  Using defaults: width={w}m, length={h}m")

    world = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    image = np.array(pts, dtype=np.float32)
    H, _ = cv2.findHomography(image, world)

    if H is None:
        print("ERROR: homography computation failed.")
        return None, 0, 0

    print("  Projection check:")
    for ip, wp in zip(pts, world):
        p = H @ np.array([*ip, 1.0])
        p = p[:2] / p[2]
        print(f"    pixel {ip} -> {p[0]:.2f}, {p[1]:.2f} m  (expected {wp[0]:.1f}, {wp[1]:.1f} m)")
    return H.tolist(), w, h, pts


# ---------------------------------------------------------------------------
# Step 2: ROI polygon
# ---------------------------------------------------------------------------

def _click_roi(img_rgb):
    """Click N points around the drivable area. Right-click / key 'q' to close."""
    pts = []

    def onclick(event):
        if event.inaxes is None:
            return
        if event.button == 3:  # right-click -> close polygon
            plt.close()
            return
        if event.button != 1:
            return
        pts.append((int(event.xdata), int(event.ydata)))
        ax.plot(event.xdata, event.ydata, "yo", markersize=5)
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "y-", linewidth=1.5)
        plt.draw()

    def onkey(event):
        if event.key == "q":
            plt.close()

    fig, ax = _show_image(img_rgb,
        "STEP 2 (optional): Click points around the drivable road area\n"
        "Left-click to add points, Right-click or press 'q' to close polygon\n"
        "If you don't need an ROI, just close the window immediately.")
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    if len(pts) < 3:
        print("  No ROI polygon defined (or fewer than 3 points). Skipping.")
        return None

    print(f"  ROI polygon: {len(pts)} points.")
    return pts


# ---------------------------------------------------------------------------
# Step 3: Speed trap lines
# ---------------------------------------------------------------------------

def _click_line(img_rgb, label, colour):
    """Click 2 points for a speed trap line. Return [(x1,y1),(x2,y2)] or None."""
    pts = []

    def onclick(event):
        if event.inaxes is None or len(pts) >= 2:
            return
        if event.button != 1:
            return
        pts.append((int(event.xdata), int(event.ydata)))
        ax.plot(event.xdata, event.ydata, "o", color=colour, markersize=7)
        if len(pts) == 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=colour, linewidth=3)
            plt.draw()
            plt.pause(0.3)
            plt.close()
            return
        plt.draw()

    fig, ax = _show_image(img_rgb,
        f"STEP 3: Click 2 points for speed trap {label}\n"
        f"Line colour = {colour}. Click two endpoints.\n"
        "Close window to skip.")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return pts if len(pts) == 2 else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def calibrate_from_image(image_path: str | Path) -> dict | None:
    """Run all calibration steps and return combined dict."""
    img = _load_image(image_path)
    if img is None:
        return None

    calib = {}

    # --- Step 1: Homography ---
    print("\n=== STEP 1: Homography calibration ===")
    result = _click_homography(img)
    if result is None:
        return None
    H, w_m, h_m, pts_uv = result
    calib["H"] = H
    calib["width_m"] = w_m
    calib["length_m"] = h_m
    calib["points_uv"] = pts_uv

    # --- Step 2: ROI polygon ---
    print("\n=== STEP 2: ROI polygon (optional) ===")
    roi = _click_roi(img)
    calib["roi_polygon"] = roi

    # --- Step 3: Speed trap lines ---
    print("\n=== STEP 3: Speed trap lines (optional) ===")
    line_a = _click_line(img, "A (first  line, upstream)", "cyan")
    line_b = _click_line(img, "B (second line, downstream)", "magenta")

    trap = None
    if line_a and line_b:
        try:
            d = float(input("  Distance between the two lines (metres): ") or "10.0")
        except (ValueError, EOFError):
            d = 10.0
            print(f"  Using default: {d}m")
        trap = {"line_a": line_a, "line_b": line_b, "distance_m": d}
        print(f"  Speed trap: {d}m between lines, {len(line_a)} + {len(line_b)} points")
    else:
        print("  Speed trap skipped (need both lines).")

    calib["speed_trap"] = trap
    calib["speed_unit"] = "mph"

    return calib


def calibrate_from_video(video_path: str | Path, frame_offset: int = 0) -> dict | None:
    """Grab a frame and run calibration."""
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(frame_offset):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: cannot read frame {frame_offset}")
        return None
    tmp = Path("/tmp/_calib_frame.jpg")
    cv2.imwrite(str(tmp), frame)
    result = calibrate_from_image(tmp)
    tmp.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Traffic camera calibrator")
    ap.add_argument("--video", help="Path to traffic video")
    ap.add_argument("--image", help="Path to reference image")
    ap.add_argument("--output", "-o", default="calibration.json")
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame offset in video to use as reference")
    args = ap.parse_args()

    if args.image:
        calib = calibrate_from_image(args.image)
    elif args.video:
        calib = calibrate_from_video(args.video, args.frame)
    else:
        ap.print_help()
        return

    if calib:
        Path(args.output).write_text(json.dumps(calib, indent=2))
        print(f"\nCalibration saved to {args.output}")


if __name__ == "__main__":
    main()
