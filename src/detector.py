from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch, cv2, numpy as np

class VLMDetector:
    """
    Vision-Language object detector wrapper.
    Call with a BGR frame and list[str] prompts → list[dict].
    """
    def __init__(self, model_name: str):
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model     = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, frame_bgr, prompts):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.processor(text=prompts, images=rgb, return_tensors="pt")
        out = self.model(**inp)
        target_sizes = torch.tensor([rgb.shape[:2]])
        results = self.processor.post_process(out, target_sizes)[0]

        dets = []
        for score, label, box in zip(results["scores"],
                                     results["labels"],
                                     results["boxes"]):
            xmin, ymin, xmax, ymax = box.int().tolist()
            dets.append({
                "prompt": prompts[label],
                "score":  float(score),
                "bbox":   np.array([xmin, ymin, xmax, ymax], np.int32)
            })
        return dets
