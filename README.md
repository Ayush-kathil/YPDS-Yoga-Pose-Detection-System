# YPDS - Yoga Pose Detection System

A real-time yoga pose detection and analysis system using computer vision and
machine learning. Get instant feedback on your yoga poses with accuracy scoring
and correction guidance.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.x-purple.svg)

## ✨ Features

- **Real-Time Pose Detection** - Uses MediaPipe's pose estimation for accurate
  body landmark tracking
- **5 Yoga Poses Supported** - Tree, Warrior II, Plank, Downdog, and Goddess
  poses
- **Accuracy Scoring** - Get a real-time accuracy score (0-100%) for your pose
- **Visual Skeleton Overlay** - See your detected pose landmarks overlaid on the
  video feed
- **Reference Images** - View reference images for each pose to guide your
  practice
- **Modern Dark UI** - Beautiful, animated interface with splash screen
- **Pose-Specific Models** - Individual ML models trained for each pose for
  higher accuracy (95% target)
- **Landmark Smoothing** - EMA-based smoothing for stable pose detection

## 📁 Project Structure

```
YPDS/
├── application/
│   ├── yoga_app.py              # Main application with UI
│   ├── yoga_poses.csv           # Pose definitions with landmark angles
│   ├── tree_accuracy_model.pkl  # Tree pose ML model
│   ├── warrior2_accuracy_model.pkl
│   ├── plank_accuracy_model.pkl
│   ├── downdog_accuracy_model.pkl
│   └── goddess_accuracy_model.pkl
├── scripts/
│   ├── generate_pose_data.py    # Generate training data
│   ├── train_accuracy_model.py  # Train pose models
│   └── yoga_poses_generated.csv # Generated pose data
├── yoga_pose_model.pkl          # Main classification model
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayush-kathil/YPDS-Yoga-Pose-Detection-System.git
   cd YPDS-Yoga-Pose-Detection-System
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe customtkinter pillow numpy scikit-learn pandas
   ```

3. **Run the application**
   ```bash
   cd application
   python yoga_app.py
   ```

## 🎯 Supported Poses

| Pose                                  | Description                                        |
| ------------------------------------- | -------------------------------------------------- |
| 🌳 **Tree (Vrikshasana)**             | Standing balance pose with one foot on inner thigh |
| ⚔️ **Warrior II (Virabhadrasana II)** | Standing pose with wide stance and arms extended   |
| 🧱 **Plank**                          | Core strengthening pose, body in straight line     |
| 🐕 **Downdog (Adho Mukha Svanasana)** | Inverted V-shape pose                              |
| 🏛️ **Goddess (Utkata Konasana)**      | Wide-legged squat with arms raised                 |

## 🔧 How It Works

1. **Pose Detection** - MediaPipe detects 33 body landmarks in real-time
2. **Feature Extraction** - Calculates joint angles (knees, hips, elbows,
   shoulders)
3. **Pose Classification** - Random Forest classifier identifies the yoga pose
4. **Accuracy Scoring** - Compares your angles against reference pose angles
5. **Visual Feedback** - Skeleton overlay shows correct (green) vs incorrect
   (red) joints

## 🧠 Technical Details

- **Pose Estimation**: MediaPipe Pose (33 landmarks)
- **ML Model**: Random Forest Classifier (150 estimators)
- **Feature Set**: 8 joint angles (bilateral symmetry)
- **Smoothing**: Exponential Moving Average (α=0.4)
- **UI Framework**: CustomTkinter with dark theme

## 📊 Training Your Own Models

1. Generate pose data:
   ```bash
   cd scripts
   python generate_pose_data.py
   ```

2. Train accuracy models:
   ```bash
   python train_accuracy_model.py
   ```

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add support for new yoga poses
- Improve pose detection accuracy
- Enhance the UI/UX
- Fix bugs and issues

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Developer

**Ayush Gupta**,
**Saksham Gupta**,
**Dhanashri Dhatrak**,
**Shikha Singh**,
**Bhakti Ramawat**,
**Bhavna Varma**

---

_Practice yoga with precision. Namaste! 🙏_


