import cv2, torch
import numpy as np

class MiDaS:
    """Intel-ISL MiDaS wrapper producing inverse-depth (bigger = nearer)."""
    def __init__(self, model_type="MiDaS_small", optimize=True):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

        if optimize and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last).half()

    @torch.inference_mode()
    def __call__(self, bgr_img: np.ndarray) -> np.ndarray:
        inp = self.transform(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)).to(self.device)
        print(type(inp))
        if self.device.type == "cuda":
            inp = inp.half()
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=bgr_img.shape[:2],
            mode="bicubic",
            align_corners=False).squeeze().cpu().numpy()
        return pred
