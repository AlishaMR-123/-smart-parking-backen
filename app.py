# server/venv/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os, io, json
import numpy as np
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))


# YOLOv8 loader
from ultralytics import YOLO as YOLOv8
# YOLOv5 loader (DetectMultiBackend from yolov5 repo)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

app = Flask(__name__)
CORS(app)

# Load models
yolov8_model = YOLOv8("./models/final_best-tara.pt")
yolov5_model = DetectMultiBackend("./models/final_best_kuek.pt", device='cpu')

# Path to client public JSONs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../client/public'))

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_json(filename, data):
    path = os.path.join(BASE_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def format_hour(hour_str):
    hour = int(hour_str.split(":")[0])
    if hour == 0:
        return "12am"
    elif hour == 12:
        return "12pm"
    elif hour < 12:
        return f"{hour}am"
    else:
        return f"{hour - 12}pm"

def predict_with_yolov5(img_pil):
    import cv2
    from yolov5.utils.augmentations import letterbox

    # Convert PIL to OpenCV image (BGR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Resize using letterbox
    img_resized = letterbox(img_cv, new_shape=640)[0]
    img_transposed = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_normalized = np.ascontiguousarray(img_transposed, dtype=np.float32) / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(yolov5_model.device)

    # Inference
    pred = yolov5_model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Return count
    return len(pred[0]) if pred[0] is not None else 0


def predict_with_yolov8(img_pil):
    results = yolov8_model(img_pil)
    return len(results[0].boxes)

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(io.BytesIO(request.files['image'].read()))
    location = request.form['location']
    date = request.form['date']
    time = request.form['time']
    model_choice = request.form.get('model', 'YOLOv8')

    if model_choice == 'YOLOv5':
        count = predict_with_yolov5(image)
    else:
        count = predict_with_yolov8(image)

    # Metadata formatting
    is_outdoor = 'outdoor' in location.lower()
    loc_type = 'outdoor' if is_outdoor else 'Indoor'
    time_key = format_hour(time)
    loc_name = location.split('-')[-1].strip() if '-' in location else location
    date_key = date

    # Update popular_times JSON
    pt_filename = f"popular_times_{loc_type}.json"
    pt_data = load_json(pt_filename)
    pt_data.setdefault(date_key, {}).setdefault(time_key, 0)
    pt_data[date_key][time_key] += count
    save_json(pt_filename, pt_data)

    # Update combined occupancy
    combined_data = load_json("combined_occupancy.json")
    combined_data.setdefault(date_key, []).append({
        "location_type": loc_type,
        "location": loc_name,
        "time": time_key,
        "vehicle_count": count
    })

    combined_data.setdefault("popular_times", {}).setdefault(loc_type, {}).setdefault(time_key, 0)
    combined_data["popular_times"][loc_type][time_key] += count
    save_json("combined_occupancy.json", combined_data)

    return jsonify({"success": True, "count": count})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
