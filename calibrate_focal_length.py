#!/usr/bin/env python
"""
Chessboard-based monocular metric-scale calibration.

1. Print a 9×6 checkerboard (default) with known square size (mm).
2. Hold it flat at a known distance (metres) from the MacBook camera.
3. Run this script – saves calibration.yaml with
     fx, fy, scale, shift  (for MiDaS inverse-depth → metres)
"""
import argparse, yaml, cv2, logging, numpy as np
from pathlib import Path
from src.camera import MacCam
from src.utils  import configure_logging

configure_logging()
log = logging.getLogger("calibration")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--square",   type=float, default=24,
                   help="checker square size in millimetres")
    p.add_argument("--board",    type=str,   default="9x6",
                   help="columns x rows (inner corners)")
    p.add_argument("--distance", type=float, default=0.40,
                   help="board distance from camera in metres")
    return p.parse_args()

def main():
    args = parse_args()
    cols, rows = map(int, args.board.lower().split("x"))
    pattern = (cols, rows)
    cam = MacCam()
    objp = np.zeros((cols*rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square / 1000.0  # mm → metres

    log.info("Press SPACE to capture a frame with a full visible board…")
    while True:
        frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("calib", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:   # SPACE
            ok, corners = cv2.findChessboardCorners(gray, pattern, None)
            if not ok:
                log.warning("Checkerboard not found – try again")
                continue
            cv2.drawChessboardCorners(frame, pattern, corners, ok)
            cv2.imshow("calib", frame)
            cv2.waitKey(500)
            break

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners], gray.shape[::-1], None, None)

    fx, fy = mtx[0, 0], mtx[1, 1]
    log.info("Estimated fx=%.1f, fy=%.1f px", fx, fy)

    # ------- depth-scale fit (simple) ----------------------------------------
    # Capture one frame for MiDaS
    from src.depth import MiDaS
    depth_net = MiDaS("DPT_Hybrid", optimize=True)
    inv_depth = depth_net(frame)        # inverse depth (bigger == nearer)
    mask = np.zeros_like(inv_depth, bool)
    cv2.drawChessboardCorners(mask.view(np.uint8), pattern, corners, 1)
    raw = inv_depth[mask]               # MiDaS values on the board

    d_med = np.median(raw)
    scale = args.distance               # z_true = scale / (d̃ + shift)
    shift = -d_med

    Path("calibration.yaml").write_text(
        yaml.safe_dump({"fx": float(fx), "fy":float(fy),
                        "scale": float(scale), "shift": float(shift)}))
    log.info("Calibration saved to calibration.yaml")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
