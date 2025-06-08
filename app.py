import gradio as gr
from PIL import Image
import numpy as np
import torch
import os, sys, cv2
from urllib.parse import urlparse, parse_qs

# Add YOLOv5 path
sys.path.append("yolov5")
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox
from ultralytics import YOLO as YOLOv8

device = torch.device("cpu")

# Load models
models = {
    "YOLOv8": YOLOv8("../models/final_best-tara.pt"),
    "YOLOv5": DetectMultiBackend("../models/final_best_kuek.pt", device=device)
}

def extract_model_from_request(request):
    try:
        referer = request.headers.get("referer", "")
        parsed_url = urlparse(referer)
        query = parse_qs(parsed_url.query)
        return query.get("model", ["YOLOv8"])[0]
    except:
        return "YOLOv8"

def predict(image, request: gr.Request):
    model_choice = extract_model_from_request(request)

    if model_choice == "YOLOv8":
        result = models["YOLOv8"](image)
        plotted = result[0].plot()
        return Image.fromarray(plotted.astype(np.uint8))

    elif model_choice == "YOLOv5":
        img = np.array(image)
        img_resized = letterbox(img, new_shape=640)[0]
        img_transposed = img_resized.transpose((2, 0, 1))
        img_normalized = np.ascontiguousarray(img_transposed, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(device)

        pred = models["YOLOv5"](img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

        # Draw boxes manually
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return Image.fromarray(img)

# Launch Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLO Vehicle Detection",
    description="Upload an image."
)

if __name__ == "__main__":
    iface.launch(share=True)
