"""
Interactive homography + ROI calibration wizard.

Usage:
    python3 traffic_intel/calibrate.py --image ref.jpg --output calib.json

Click 4 road corners -> press 'h' -> bird's-eye preview -> 's' to save.
World coords auto-assigned: odd-index clicks get x=lane-width, clicks 3+ get y=segment-length.
Adjust defaults via --lane-width and --segment-length.

Controls:
  Left-click       Add point
  Right-click      Remove last point
  h                Compute homography + prompt for world coords
  r                Toggle ROI / Homography mode
  c                Clear points
  s                Save to file
  q                Quit
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("gtk4agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


LANE_WIDTH = 3.7
SEGMENT_LEN = 25.0

def main():
    ap = argparse.ArgumentParser(description="Calibration wizard")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="calib.json")
    ap.add_argument("--lane-width", type=float, default=LANE_WIDTH)
    ap.add_argument("--segment-length", type=float, default=SEGMENT_LEN)
    ap.add_argument("--pixels-per-meter", type=float, default=30,
                    help="Output resolution for bird's-eye view (default: 30)")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"Cannot load {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, (ax_main, ax_bird) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]},
    )
    ax_main.imshow(img_rgb)
    ax_bird.set_title("Bird's-eye view")
    fig.tight_layout()

    h_pts: list = []
    w_pts: list = []
    roi_pts: list = []
    roi_mode = False
    H = None
    warped_rgb = None

    def redraw():
        nonlocal H, warped_rgb
        for l in ax_main.lines + ax_main.texts + list(ax_main.patches):
            l.remove()
        if len(roi_pts) > 2:
            ax_main.add_patch(Polygon(roi_pts, fill=False, edgecolor="yellow", linewidth=2))
        for p in roi_pts:
            ax_main.plot(p[0], p[1], "yo", markersize=5)
        for i, (x, y) in enumerate(h_pts):
            c = "green" if i < 4 else "orange"
            ax_main.plot(x, y, "o", color=c, markersize=7)
            ax_main.text(x + 6, y - 6, str(i + 1), color=c, fontsize=10, weight="bold")
            if i < len(w_pts):
                wx, wy = w_pts[i]
                ax_main.text(x + 6, y + 10, f"{wx:.1f},{wy:.1f}m", color="lightblue", fontsize=8)
        mode = "ROI" if roi_mode else "HOMOGRAPHY"
        n = len(roi_pts if roi_mode else h_pts)
        ax_main.set_title(f"Mode: {mode} | {n} pts")
        ax_bird.cla()
        if warped_rgb is not None:
            ax_bird.imshow(warped_rgb)
        ax_bird.set_title("Bird's-eye view")
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal roi_mode, H, warped_rgb
        if event.inaxes != ax_main or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if event.button == 1:
            if roi_mode:
                roi_pts.append((x, y))
                print(f"ROI {len(roi_pts)}: ({x},{y})")
            else:
                h_pts.append((x, y))
                print(f"Point {len(h_pts)}: ({x},{y})")
            redraw()
        elif event.button == 3:
            if roi_mode and roi_pts:
                roi_pts.pop()
            elif not roi_mode and h_pts:
                h_pts.pop()
                if w_pts:
                    w_pts.pop()
                H, warped_rgb = None, None
            redraw()

    def on_key(event):
        nonlocal roi_mode, H, warped_rgb
        k = event.key.lower()
        if k == "q":
            plt.close()
        elif k == "r":
            roi_mode = not roi_mode
            print(f"{'ROI' if roi_mode else 'Homography'} mode")
            redraw()
        elif k == "c":
            h_pts.clear()
            w_pts.clear()
            roi_pts.clear()
            H, warped_rgb = None, None
            redraw()
        elif k == "h" and len(h_pts) >= 4:
            # Auto-assign world coords: odd idx x=lane_width, even idx x=0
            # idx >=2 gets segment_length for y
            for i in range(len(h_pts)):
                if i < len(w_pts):
                    continue
                wx = args.lane_width if i % 2 == 1 else 0.0
                wy = args.segment_length if i >= 2 else 0.0
                w_pts.append((wx, wy))
                print(f"  Pt {i+1} world ({wx:.2f}, {wy:.2f}) m (--lane-width {args.lane_width}, --segment-length {args.segment_length})")
            n = min(len(h_pts), len(w_pts))
            if n >= 4:
                ip = np.float32(h_pts[:n])
                wp = np.float32(w_pts[:n])
                if n == 4:
                    H = cv2.getPerspectiveTransform(ip, wp)
                else:
                    H, _ = cv2.findHomography(ip, wp, cv2.RANSAC, 5.0)
                if H is not None:
                    print(f"Homography:\n{H}")
                    ppm = args.pixels_per_meter
                    out_w = int(args.lane_width * ppm) or 1
                    out_h = int(args.segment_length * ppm) or 1
                    # Scale H so world (0,0) maps to pixel (0,0)
                    # and world (lw, sl) maps to pixel (lw*ppm, sl*ppm)
                    S = np.array([[ppm, 0, 0], [0, ppm, 0], [0, 0, 1]], dtype=np.float32)
                    warp = cv2.warpPerspective(img_bgr, S @ H, (out_w, out_h))
                    warped_rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            redraw()
        elif k == "s":
            data = dict(
                image_points=h_pts,
                world_points=w_pts,
                roi_polygon=roi_pts,
                homography_matrix=H.tolist() if H is not None else [],
            )
            Path(args.output).write_text(json.dumps(data, indent=2))
            print(f"Saved to {args.output}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("READY")
    plt.show()


if __name__ == "__main__":
    main()
