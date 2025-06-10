import cv2, logging
class MacCam:
    """Tiny OpenCV wrapper to grab frames at a fixed resolution."""
    def __init__(self, id: int = 0, resolution_h=720, resolution_w=1280, fov_deg=69, **_):
        self.cap = cv2.VideoCapture(id, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam")
        self.fov_deg = fov_deg
        logging.getLogger("camera").info("Camera ready at %s, %s px", resolution_w, resolution_h)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failure")
        return frame

    def release(self):
        self.cap.release()
