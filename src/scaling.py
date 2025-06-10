import yaml, numpy as np
from pathlib import Path

class DepthScaler:
    """
    Converts MiDaS inverse-depth to metric metres, using either:
      * saved calibration.yaml               (scale & shift), or
      * fallback FoV pinhole approximation   (rough).
    """
    def __init__(self, calib_path="calibration.yaml", cfg_path="config.yaml"):
        self.use_calib = Path(calib_path).exists()
        if self.use_calib:
            data = yaml.safe_load(open(calib_path))
            self.scale, self.shift = data["scale"], data["shift"]
        else:
            cfg = yaml.safe_load(open(cfg_path))
            w = cfg["camera"]["resolution_w"]
            fov = np.deg2rad(cfg["camera"]["fov_deg"])
            fx = (w / 2) / np.tan(fov / 2)
            self.scale, self.shift = fx, 0.0   # very approximate

    def estimate(self, inv_depth: np.ndarray, bbox):
        x1, y1, x2, y2 = bbox
        roi = inv_depth[y1:y2, x1:x2]
        raw = np.median(roi)
        return self.scale / (raw + self.shift)
    
    def fit_single(self, inv_depth_value: float, true_cm: float):
        """
        Update scale from one (inverse-depth, true-distance) pair.
        Uses   z = scale / d̃   with fixed shift = 0.
        """
        self.shift = 0.0
        self.scale = (true_cm / 100.0) * inv_depth_value      # metres
        print(f"[calib] set scale={self.scale:.4f}, shift=0")
        Path("calibration.yaml").write_text(
            yaml.safe_dump({"scale": float(self.scale), "shift": 0.0}))
