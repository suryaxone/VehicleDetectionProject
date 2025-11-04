# 🚦 Vehicle Detection and Red Light Violation Challan System

## 📋 Overview

This project automatically detects vehicles in a traffic surveillance video, identifies **red light violations**, and generates a **PDF challan** for each violating vehicle.
It uses **YOLOv5** for real-time object detection and **ReportLab** for generating professional PDF challans with embedded vehicle images.

---

## 🧠 Features

* Detects vehicles such as **cars**, **buses**, and **trucks** using YOLOv5.
* Simulates a **red-light traffic signal** in the video.
* Detects vehicles that **cross the red line** during the red signal.
* Saves cropped images of violating vehicles.
* Generates an **auto-filled challan PDF** containing:

  * Violation ID
  * Driver/Owner Name (sample)
  * Vehicle Number (sample)
  * Fine Amount
  * Date & Time
  * Vehicle Image

---

## 🗂️ Folder Structure

```
Vehicle-Detection-Project/
│
├── main.py                # Main program file
├── best.pt                # YOLOv5 model weights
├── traffic_video.mp4      # Input video file
│
├── yolov5/                # YOLOv5 model source files
├── challans/              # Auto-generated challan PDFs
└── violations/            # Cropped vehicle violation images
```

---

## ⚙️ Technologies Used

| Library             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| **PyTorch (torch)** | Runs the YOLOv5 model for object detection |
| **OpenCV (cv2)**    | Reads and processes video frames           |
| **NumPy**           | Handles array operations                   |
| **ReportLab**       | Creates and formats PDF challans           |
| **Datetime**        | Generates timestamps for each violation    |
| **OS**              | Handles file and folder operations         |

---

## 🚀 How It Works

1. The system reads the input traffic video.
2. YOLOv5 detects vehicles frame by frame.
3. A **red signal** is simulated every few seconds.
4. If a vehicle crosses the **red line** while the signal is red:

   * The vehicle’s image is cropped and saved.
   * A **PDF challan** is generated and saved in the `challans/` folder.
5. The process continues until the video ends.

---

## 🧩 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Vehicle-Detection-Project.git
cd Vehicle-Detection-Project
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install reportlab
```

### 3️⃣ Download YOLOv5 Model

```bash
curl -L -o best.pt https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

*(Or download manually and place it in your project folder.)*

---

## ▶️ Run the Program

```bash
python main.py
```

* Press **Q** to stop the video preview.
* Challan PDFs will appear inside `challans/`.
* Cropped vehicle images appear in `violations/`.

---

## 🧾 Sample Challan PDF

Each generated PDF contains:

* Violation details (ID, name, number plate, date)
* Fine amount
* Captured vehicle image

---

## 👥 Collaboration

To collaborate:

1. Go to your GitHub repository → **Settings → Collaborators**
2. Add teammates by GitHub username.
3. They can clone and push updates using:

   ```bash
   git pull
   git push
   ```

### 🧩 Future Enhancements

* Automatic license plate recognition (ANPR)
* Database integration for storing challan records
* Web dashboard for monitoring violations
