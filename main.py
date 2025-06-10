import yaml, cv2, logging
import numpy as np
from pathlib import Path

from src.camera   import MacCam
from src.detector import VLMDetector
from src.depth    import MiDaS
from src.scaling  import DepthScaler
from src.ui       import Overlay, ClickSelector
from src.utils    import configure_logging

configure_logging()
logger = logging.getLogger("distance-cam")

cfg = yaml.safe_load(open("config.yaml"))
cam       = MacCam(**cfg["camera"])
detector  = VLMDetector(cfg["vlm"]["model"])
depth_net = MiDaS(cfg["depth"]["model_type"], cfg["depth"]["optimize"])
scaler    = DepthScaler()      # auto-loads calibration.yaml if present
selector  = ClickSelector(cfg["ui"]["click_radius_px"])

prompt = [input("Describe the object to track (e.g. 'coffee mug'): ").strip()]
if not prompt[0]:
    raise SystemExit("Prompt cannot be empty")

logger.info("Click inside a bounding-box to choose another instance…")

cv2.namedWindow("distance-cam")
cv2.setMouseCallback("distance-cam", selector)


try:
    selector  = ClickSelector(cfg["ui"]["click_radius_px"])

    PAUSED = False
    while True:
        frame = cam.read()
        if not PAUSED:
            dets  = detector(frame, prompt)
            depth = depth_net(frame)

        if dets:
            idx  = selector.select(dets) or 0
            bbox = dets[idx]["bbox"]
            dist = scaler.estimate(depth, bbox)
            Overlay.draw(frame, [dets[idx]], [dist])
            # write dist on the framw
            cv2.putText(frame, f"Distance: {dist:.2f} m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("distance-cam", frame)
        key = cv2.waitKey(1) & 0xFF
        print(f"Key pressed: {key}")

        if key == 27:                       # ESC – quit
            break
        elif key == ord('p'):               # pause / resume
            PAUSED = not PAUSED
        elif key == ord('c') and PAUSED:    # calibrate on paused frame
            pt = selector.clicked_point()
            if pt is None:
                print("Click an object first, then press 'c'")
                continue
            # find bbox whose centre is closest to last click
            idx = selector.select(dets) or 0
            bbox = dets[idx]["bbox"]
            raw  = np.median(depth[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            true_cm = float(input("Enter true distance to object (cm): ").strip())
            scaler.fit_single(raw, true_cm)
finally:
    cam.release()
    cv2.destroyAllWindows()
