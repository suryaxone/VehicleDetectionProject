import os
import sys
import cv2
import torch
import numpy as np
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATION_FOLDER = os.path.join(BASE_DIR, "violations")
CHALLAN_FOLDER = os.path.join(BASE_DIR, "challans")
YOLOV5_FOLDER = os.path.join(BASE_DIR, "yolov5")

os.makedirs(VIOLATION_FOLDER, exist_ok=True)
os.makedirs(CHALLAN_FOLDER, exist_ok=True)
sys.path.append(YOLOV5_FOLDER)

# === IMPORT YOLO COMPONENTS ===
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# === UTILITIES ===
def sanitize_filename(name: str) -> str:
    """Removes illegal filename chars for Windows."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

def create_challan_pdf(violation):
    """Creates a challan PDF for the violator."""
    safe_id = sanitize_filename(violation["id"])
    pdf_path = os.path.join(CHALLAN_FOLDER, f"{safe_id}.pdf")

    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 2.5 * cm, "TRAFFIC VIOLATION CHALLAN")

        c.setFont("Helvetica", 12)
        y = height - 4 * cm
        for key, value in [
            ("Challan ID", violation["id"]),
            ("Vehicle Type", violation["vehicle_type"]),
            ("Vehicle Number", violation["number_plate"]),
            ("Offense", "Red Light Violation"),
            ("Fine", f"₹{violation['fine']}"),
            ("Date & Time", violation["date"]),
            ("Location", violation["location"]),
        ]:
            c.drawString(3 * cm, y, f"{key}: {value}")
            y -= 1 * cm

        # Attach violator image
        if os.path.exists(violation["image_path"]):
            img_w, img_h = 12 * cm, 8 * cm
            c.drawImage(violation["image_path"], 3 * cm, y - img_h, width=img_w, height=img_h)
        else:
            print(f"⚠️ Image not found: {violation['image_path']}")

        c.setFont("Helvetica-Oblique", 10)
        c.drawString(3 * cm, 2 * cm, "System-generated challan (no signature required).")
        c.save()
        print(f"✅ Challan generated: {pdf_path}")

    except Exception as e:
        print(f"❌ Error generating challan: {e}")

# === LOAD YOLO MODEL ===
model_path = os.path.join(BASE_DIR, "best.pt")
if not os.path.exists(model_path):
    print(f"❌ YOLO model not found at: {model_path}")
    sys.exit()

print("🔄 Loading YOLOv5 model...")
model = DetectMultiBackend(model_path)
model.eval()
stride, names = model.stride, model.names
img_size = 640
print("✅ YOLOv5 model loaded successfully.\n")

# === LOAD VIDEO ===
videos = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".mp4")]
if not videos:
    print("❌ No .mp4 file found in folder.")
    sys.exit()

video_path = os.path.join(BASE_DIR, videos[0])
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    sys.exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps
print(f"🎥 Loaded video: {os.path.basename(video_path)} ({video_duration:.2f}s)\n")

# === DETECTION ===
print("🚦 Starting vehicle detection (one challan per vehicle)...\n")

violated = set()  # to prevent multiple challans for same vehicle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps

    # === YOLO INFERENCE ===
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cls_name = names[int(cls)] if int(cls) in names else "vehicle"

                # Create a simple hash-like ID for the vehicle bounding box
                vid = f"{cls_name}_{x1}_{y1}_{x2}_{y2}"

                # If not already processed, create challan
                if vid not in violated:
                    violated.add(vid)
                    print(f"🚨 New vehicle detected: {cls_name}")

                    # Save cropped vehicle image
                    crop = frame[y1:y2, x1:x2]
                    violation_id = f"V_{cls_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    image_path = os.path.join(VIOLATION_FOLDER, f"{sanitize_filename(violation_id)}.jpg")
                    cv2.imwrite(image_path, crop)

                    # Generate challan
                    violation_info = {
                        "id": violation_id,
                        "name": "Unknown Driver",
                        "number_plate": "MH12AB1234",
                        "vehicle_type": cls_name,
                        "fine": 2000,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": image_path,
                        "location": "Signal Junction, City Center"
                    }
                    create_challan_pdf(violation_info)

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection complete.")
print(f"📸 Vehicle images saved in: {VIOLATION_FOLDER}")
print(f"📄 Challans saved in: {CHALLAN_FOLDER}")