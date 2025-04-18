# 🧠 Face Recognition System on Raspberry Pi

This project is a lightweight, real-time face detection and recognition system built using **OpenCV** and deployed on a **Raspberry Pi 4 Model B**. It detects faces using Haar Cascades and recognizes them using the **LBPH** (Local Binary Patterns Histogram) algorithm.

---

## 🚀 Features

- Face detection using Haar Cascade
- Face recognition using OpenCV's LBPH recognizer
- Training script to register new users
- Labels and model persistence using Pickle and YAML
- Unknown face handling
- Real-time recognition via USB camera
- Works entirely offline on Raspberry Pi

---

## 🧰 Requirements

- Raspberry Pi 4 (or any model with decent performance)
- USB webcam
- Python 3.7+
- OpenCV (`opencv-contrib-python`)
- VNC setup (optional for GUI access)

---

## 📂 Project Structure

```
├── capture.py         # Captures face images and stores them in folders
├── train2.py           # Trains the LBPH model and saves it
├── recog3.py       # Real-time recognition using webcam
├── labels.pickle      # Stores label-name mappings
├── trainer.yml        # Trained recognizer model
├── haarcascade_frontalface_default.xml
└── dataset/           # Directory where face images are stored
    ├── Gautam/
    ├── Ankit/
    └── Priya/
```

---

## 📸 How to Use

### 1. Capture Face Images

```bash
python3 capture.py
```

Enter your name and let the system capture 30–50 images.

### 2. Train the Recognizer

```bash
python3 train2.py
```

This creates `trainer.yml` and `labels.pickle`.

### 3. Run Face Recognition

```bash
python3 recog3.py
```

It opens the camera feed and labels known faces in real-time.

---

## 📝 Notes

- Place `haarcascade_frontalface_default.xml` in the same directory or provide a valid path.
- Ensure good lighting and proper camera alignment during image capture for better accuracy.
- Unknown faces are labeled `"Unknown"` automatically if no match is found within a confidence threshold.

---

## 📦 Installation (on Raspberry Pi)

```bash
sudo apt update
sudo apt install python3-opencv python3-pip
pip3 install opencv-contrib-python --break-system-packages
```

Or better, use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-contrib-python
```

---

## 📄 License

This project is open-source and free to use under the MIT License.

---

## 👤 Author

**Sanskar Koserwal**  
Built as part of a smart attendance/surveillance system prototype using Raspberry Pi.
