"""
Interactive homography calibrator.

Click 4 points on a reference frame that form a rectangle on the road
(in TL/TR/BR/BL order), enter real-world dimensions, and save the
homography matrix to a JSON file.

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


def _click_callback(pts):
    """Return a plt callback that appends clicks up to 4."""
    def on_click(event):
        if event.inaxes is None or len(pts) >= 4:
            return
        pts.append((int(event.xdata), int(event.ydata)))
        ax = event.inaxes
        ax.plot(event.xdata, event.ydata, "ro", markersize=6)
        ax.text(event.xdata + 5, event.ydata - 5, str(len(pts)),
                fontsize=12, color="red", fontweight="bold")
        if len(pts) >= 2:
            # draw lines
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "r-", linewidth=1)
        plt.draw()
    return on_click


def calibrate_from_image(image_path: str | Path) -> dict | None:
    """Launch interactive calibration and return calibration dict, or None."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: cannot read {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pts = []
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_rgb)
    ax.set_title(
        "Click 4 points on the road in ORDER:\n"
        "1=Top-Left  2=Top-Right  3=Bottom-Right  4=Bottom-Left\n"
        "(they should form a rectangle on the road surface)\n"
        "Close window when done."
    )
    fig.canvas.mpl_connect("button_press_event", _click_callback(pts))
    plt.tight_layout()
    plt.show()

    if len(pts) < 4:
        print("Need exactly 4 points. Exiting.")
        return None

    # Ask for real-world dimensions
    print(f"\nSelected points (image coords): {pts}")
    try:
        w = float(input("Road WIDTH (metres, perpendicular to traffic): ") or "3.7")
        h = float(input("Road LENGTH (metres, along traffic direction): ") or "10.0")
    except (ValueError, EOFError):
        w, h = 3.7, 10.0
        print(f"Using defaults: width={w}m, length={h}m")

    # World points: rectangle in ground plane
    world_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    image_pts = np.array(pts, dtype=np.float32)

    H, _ = cv2.findHomography(image_pts, world_pts)
    if H is None:
        print("ERROR: homography computation failed.")
        return None

    calib = dict(
        H=H.tolist(),
        width_m=w,
        length_m=h,
        points_uv=pts,
    )
    print(f"\nHomography computed. Projection test:")
    for ip, wp in zip(pts, world_pts):
        # H @ [u, v, 1]
        p = H @ np.array([*ip, 1.0])
        p = p[:2] / p[2]
        print(f"  pixel {ip} → {p[0]:.2f}, {p[1]:.2f} m  (expected {wp[0]:.1f}, {wp[1]:.1f} m)")
    return calib


def calibrate_from_video(video_path: str | Path, frame_offset: int = 0) -> dict | None:
    """Grab a frame from the video and launch calibration."""
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(frame_offset):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: cannot read frame {frame_offset} from {video_path}")
        return None
    tmp = Path("/tmp/_calib_frame.jpg")
    cv2.imwrite(str(tmp), frame)
    result = calibrate_from_image(tmp)
    tmp.unlink(missing_ok=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Traffic camera homography calibrator")
    ap.add_argument("--video", help="Path to traffic video")
    ap.add_argument("--image", help="Path to reference image")
    ap.add_argument("--output", "-o", default="calibration.json",
                    help="Output JSON path")
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
