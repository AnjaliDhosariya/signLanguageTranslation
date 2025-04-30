# 🤟 Real-Time Sign Language Translator using MediaPipe & LSTM

This project is a real-time hand gesture recognition system that translates sign language alphabets (A–Z) into text using a webcam. It uses **MediaPipe** for hand landmark detection and an **LSTM-based neural network** for gesture classification.
---

## 🚀 Features

- Real-time gesture detection using webcam
- Translates A–Z (customizable gestures)
- Uses MediaPipe for accurate hand tracking
- Deep learning model with stacked LSTM layers
- Dynamic prediction smoothing and confidence filtering
- Simple, modular Python codebase

---

## 📂 Project Structure

```
├── app.py               # Main real-time detection application
├── train.py             # Model training script
├── dataConversion.py    # Data preprocessing & landmark extraction
├── function.py          # Core helper functions (landmark, drawing, extraction)
├── model.h5             # Trained model weights
├── model.json           # Trained model architecture
├── MP_Data/             # Directory for processed training data
└── README.md
```

---

## 🧠 Requirements

- Python 3.7+
- TensorFlow (2.x)
- OpenCV
- MediaPipe
- NumPy

📦 Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

---

## 🏋️‍♂️ Training the Model

The model was trained using:
- **Categorical Crossentropy** loss function
- **Adam optimizer** for adaptive learning
- **Train-test split**: 70% training, 15% validation, 15% testing
- Achieved **92% validation accuracy**

1. Collect or prepare images for each gesture (A–Z)
2. Extract hand keypoints with `dataConversion.py`
3. Train model with:

```bash
python train.py
```

This saves:
- `model.h5` – trained model weights
- `model.json` – model structure


---

## 🎯 Running the Translator

Make sure `model.json` and `model.h5` are present in the project folder.

```bash
python app.py
```

Use your webcam to perform gestures. The recognized gesture will appear on the screen in real time.

### 📸 Example Outputs

Below are some snapshots from the real-time detection window:

| Gesture | Detection Example |
|--------|--------------------|
| L      | ![Image](https://github.com/user-attachments/assets/c831aa13-95d5-4df0-91af-9ffbe79e8e84) |
| O      | ![Image](https://github.com/user-attachments/assets/6fb47a9e-3f50-4a3c-afb5-6c2724f24bcb)|
| W      | ![Image](https://github.com/user-attachments/assets/f42befaa-775d-4bd4-90ec-02982e44501a) |



Make sure `model.json` and `model.h5` are present in the project folder.

```bash
python app.py
```

Use your webcam to perform gestures. The recognized gesture will appear on the screen in real time.

---

## ⚙️ Model Architecture

- Input: 70 frames of 21 landmarks (x, y, z) → 63 features per frame
- 2 LSTM layers (64 → 128units)
- 3 Dropout layers (0.3)
- 2 Dense layers (64 → 32 units)
- Output: Softmax over number of gestures

---

## 🔧 Customization

- ✋ Add your own gestures by editing `actions` in `function.py`
- 🎥 Adjust `sequence_length` in `app.py` for responsiveness
- 🔍 Tweak `threshold` to control confidence filtering
- 📊 Want accuracy graphs or evaluation? Add `model.evaluate()` or TensorBoard in `train.py`

---

## 🙌 Acknowledgments

- [MediaPipe](https://mediapipe.dev) for hand landmark detection
- [TensorFlow](https://tensorflow.org) for the LSTM model
- Inspired by projects like [Sign Language Detection](https://github.com/nicknochnack/Sign-Language-Detection)

---

## 📜 License

MIT License – feel free to use, modify, and share!

---

## 👋 Get in Touch

If you found this project useful or want to collaborate, feel free to reach out or star the repo 💫

