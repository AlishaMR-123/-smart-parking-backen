from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os, io, json
import numpy as np
import requests
from ultralytics import YOLO as YOLOv8

# Hugging Face model URL (YOLOv8)
YOLOV8_MODEL_URL = "https://huggingface.co/spaces/Alishaaa199/yolo-vehicle-detection/resolve/main/final_best-tara.pt"

# Download model if not present
def download_model(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path} ...")
        r = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {local_path} ✅")

# Create models directory if needed
os.makedirs("models", exist_ok=True)

# Download model
download_model(YOLOV8_MODEL_URL, "models/final_best-tara.pt")

# Load model
yolov8_model = YOLOv8("models/final_best-tara.pt")

# Init Flask
app = Flask(__name__)
CORS(app)

# Path to client public JSONs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'json'))
os.makedirs(BASE_DIR, exist_ok=True)

# JSON utils
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

# Format hour
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

# Predict with YOLOv8
def predict_with_yolov8(img_pil):
    img_pil = img_pil.resize((640, 640))  # Resize for memory efficiency ✅
    results = yolov8_model(img_pil)
    return len(results[0].boxes)

# /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(io.BytesIO(request.files['image'].read()))
    location = request.form['location']
    date = request.form['date']
    time = request.form['time']

    # Predict
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


@app.route('/popular_times_indoor.json')
def serve_popular_times_indoor():
    return send_from_directory(BASE_DIR, 'popular_times_Indoor.json')  # Note: filename uses 'Indoor' capital I!

@app.route('/popular_times_outdoor.json')
def serve_popular_times_outdoor():
    return send_from_directory(BASE_DIR, 'popular_times_outdoor.json')

@app.route('/combined_occupancy.json')
def serve_combined_occupancy():
    return send_from_directory(BASE_DIR, 'combined_occupancy.json')

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
