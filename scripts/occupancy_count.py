import os
import json
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = '../models/final_best-tara.pt'
DATASET_ROOT = '../../Data'  # Contains both 'Indoor' and 'outdoor'
OUTPUT_JSON = '../../client/public/combined_occupancy.json'
# ==============

# Load model
print(f"🔍 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

occupancy_data = []
popular_times = {}

# Traverse both Indoor and Outdoor folders
for location_type in ['Indoor', 'outdoor']:
    location_path = os.path.join(DATASET_ROOT, location_type)
    if not os.path.isdir(location_path):
        print(f"❌ Directory not found: {location_path}")
        continue

    print(f"\n📁 Scanning {location_type}...")

    for hour_folder in sorted(os.listdir(location_path)):
        hour_path = os.path.join(location_path, hour_folder)
        if not os.path.isdir(hour_path):
            print(f"⏩ Skipping non-folder: {hour_path}")
            continue

        hour_total = 0

        if location_type == 'Indoor':
            for block in os.listdir(hour_path):
                block_path = os.path.join(hour_path, block)
                if not os.path.isdir(block_path):
                    continue

                print(f"🔹 Indoor - {block} at {hour_folder}")
                for file in os.listdir(block_path):
                    if file.lower().endswith(('.jpg', '.png')):
                        img_path = os.path.join(block_path, file)
                        results = model(img_path)
                        count = len(results[0].boxes)
                        print(f"    {file} → {count} vehicles")

                        occupancy_data.append({
                            "location_type": location_type,
                            "location": block,
                            "time": hour_folder,
                            "vehicle_count": count
                        })
                        hour_total += count

        else:
            print(f"🔸 Outdoor - {hour_folder}")
            for file in os.listdir(hour_path):
                if file.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(hour_path, file)
                    results = model(img_path)
                    count = len(results[0].boxes)
                    print(f"    {file} → {count} vehicles")

                    occupancy_data.append({
                        "location_type": location_type,
                        "location": "Outdoor",
                        "time": hour_folder,
                        "vehicle_count": count
                    })
                    hour_total += count

        if location_type not in popular_times:
            popular_times[location_type] = {}

        popular_times[location_type][hour_folder] = hour_total
        print(f"📊 Total for {location_type} at {hour_folder}: {hour_total} vehicles")

# Save output
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump({
        "occupancy_data": occupancy_data,
        "popular_times": popular_times
    }, f, indent=2)

print(f"\n✅ Combined occupancy written to: {OUTPUT_JSON}")
