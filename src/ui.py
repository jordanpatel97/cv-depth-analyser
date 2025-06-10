import cv2, numpy as np

class Overlay:
    """Utility to draw bbox + distance text."""
    BOX_COLOR  = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)

    @staticmethod
    def draw(frame, dets, dists_m):
        for det, dist in zip(dets, dists_m):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), Overlay.BOX_COLOR, 2)
            lbl = f"{det['prompt']}: {dist:.2f} m"
            cv2.putText(frame, lbl, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        Overlay.TEXT_COLOR, 2, cv2.LINE_AA)
        return frame

class ClickSelector:
    """
    Mouse callback helper – remembers the bbox whose centre
    is closest to the click.
    """
    def __init__(self, radius_px=15):
        self.radius = radius_px
        self.pt = None

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt = (x, y)

    def select(self, dets):
        if self.pt is None:
            return 0
        cx, cy = self.pt
        dists = [np.hypot(cx - (b["bbox"][0]+b["bbox"][2])/2,
                          cy - (b["bbox"][1]+b["bbox"][3])/2)
                 for b in dets]
        idx = int(np.argmin(dists))
        if dists[idx] < self.radius:
            return idx
        return 0
    
    def clicked_point(self):
        """Return last click coordinates (x, y) or None."""
        return self.pt
