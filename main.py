# main.py
# -------------------------------------------------------
# VEHICLE RED LIGHT VIOLATION DETECTION & CHALLAN SYSTEM
# -------------------------------------------------------
# Detects vehicles that cross a red signal line using YOLOv5
# and automatically generates a challan PDF for each violator.
# -------------------------------------------------------

import os
import sys
import cv2
import torch
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# -------------------------------------------------------
# 1. SETUP PROJECT PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATION_FOLDER = os.path.join(BASE_DIR, "violations")
CHALLAN_FOLDER = os.path.join(BASE_DIR, "challans")
YOLOV5_FOLDER = os.path.join(BASE_DIR, "yolov5")

os.makedirs(VIOLATION_FOLDER, exist_ok=True)
os.makedirs(CHALLAN_FOLDER, exist_ok=True)

sys.path.append(YOLOV5_FOLDER)

# -------------------------------------------------------
# 2. IMPORT YOLOv5 COMPONENTS
# -------------------------------------------------------
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# -------------------------------------------------------
# 3. PDF CHALLAN GENERATION FUNCTION
# -------------------------------------------------------
def create_challan_pdf(violation):
    """
    Creates a challan PDF for a red light violation.
    """
    pdf_path = os.path.join(CHALLAN_FOLDER, f"{violation['id']}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 3*cm, "Traffic Red Light Violation Challan")

    # Details
    c.setFont("Helvetica", 12)
    y = height - 5*cm
    gap = 1*cm
    c.drawString(3*cm, y, f"Violation ID: {violation['id']}")
    y -= gap
    c.drawString(3*cm, y, f"Driver/Owner Name: {violation['name']}")
    y -= gap
    c.drawString(3*cm, y, f"Vehicle Number Plate: {violation['number_plate']}")
    y -= gap
    c.drawString(3*cm, y, f"Fine Amount: ₹{violation['fine']}")
    y -= gap
    c.drawString(3*cm, y, f"Date & Time: {violation['date']}")
    y -= gap

    # Vehicle image
    if os.path.exists(violation["image_path"]):
        img_w = 12*cm
        img_h = 12*cm
        c.drawImage(violation["image_path"], 3*cm, y - img_h, width=img_w, height=img_h)

    c.save()
    print(f"✅ Challan PDF created: {pdf_path}")

# -------------------------------------------------------
# 4. LOAD YOLOv5 MODEL
# -------------------------------------------------------
model_path = os.path.join(BASE_DIR, "best.pt")
if not os.path.exists(model_path):
    print(f"❌ ERROR: YOLO model not found at {model_path}")
    sys.exit()

print("🔄 Loading YOLOv5 model...")
model = DetectMultiBackend(model_path)
model.eval()
stride, names = model.stride, model.names
img_size = 640
print("✅ YOLOv5 model loaded successfully.")

# -------------------------------------------------------
# 5. FIND VIDEO FILE
# -------------------------------------------------------
video_candidates = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".mp4")]
if not video_candidates:
    print("❌ ERROR: No video file found in project folder.")
    sys.exit()

video_file = os.path.join(BASE_DIR, video_candidates[0])
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video {video_file}")
    sys.exit()

# -------------------------------------------------------
# 6. DETECTION PARAMETERS
# -------------------------------------------------------
frame_num = 0
signal_state = "RED"  # Simulated traffic light
signal_timer = 0
signal_duration = 150  # frames between state changes
line_y = 300  # position of the stop line (adjust as per video)

# -------------------------------------------------------
# 7. START PROCESSING VIDEO
# -------------------------------------------------------
print("\n🚦 Starting red light violation detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Toggle signal (simulate red/green every few seconds)
    signal_timer += 1
    if signal_timer > signal_duration:
        signal_state = "GREEN" if signal_state == "RED" else "RED"
        signal_timer = 0

    # Draw stop line and traffic signal on video
    color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), color, 3)
    cv2.circle(frame, (50, 50), 20, color, -1)
    cv2.putText(frame, f"Signal: {signal_state}", (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # YOLOv5 preprocessing
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # YOLO inference
    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                cls = int(cls)
                cls_name = names.get(cls, f"class_{cls}")

                # Draw detection boxes
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Check for violation
                vehicle_center_y = (y1 + y2) // 2
                if signal_state == "RED" and vehicle_center_y < line_y:
                    print(f"🚨 Violation detected at frame {frame_num} ({cls_name})")

                    # Crop violating vehicle
                    crop = frame[y1:y2, x1:x2]
                    violation_id = f"V{frame_num}_{datetime.now().strftime('%H%M%S')}"
                    image_path = os.path.join(VIOLATION_FOLDER, f"{violation_id}.jpg")
                    cv2.imwrite(image_path, crop)

                    # Generate PDF challan
                    violation_info = {
                        "id": violation_id,
                        "name": "John Doe",
                        "number_plate": "MH12AB1234",
                        "fine": 2000,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": image_path
                    }
                    create_challan_pdf(violation_info)

    # Optional: show video while processing (press 'q' to quit)
    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Detection complete.")
print(f"📸 Violation images saved in: {VIOLATION_FOLDER}")
print(f"📄 Challan PDFs saved in: {CHALLAN_FOLDER}")
