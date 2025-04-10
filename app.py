from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints, actions
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import mediapipe as mp

#Loading model
print("[INFO] Loading model...")
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.h5")
model.make_predict_function()
print("[INFO] Model loaded.")

# Simplified colors list
colors = [(245, 117, 16)] * len(actions)

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Parameters initialization
sequence = []
sentence = []
accuracy = []
predictions = []
THRESHOLD = 0.75
SEQUENCE_LENGTH = 30  # few frames for faster prediction

# Initializing webcam
print("[INFO] Starting video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

# Initializing MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("[INFO] MediaPipe Hands initialized.")

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not read properly, exiting loop.")
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[DEBUG] Processing frame #{frame_count}")

        # Define ROI frame
        crop_frame = frame[40:400, 0:300]
        image, results = mediapipe_detection(crop_frame, hands)

        if results.multi_hand_landmarks:
            print("[DEBUG] Hand detected")
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Triming sequence
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]

            # Predicting when we have enough frames
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict_on_batch(
                    np.expand_dims(sequence, axis=0)
                )[0]
                pred_class = np.argmax(res)
                predictions.append(pred_class)
                print(f"[DEBUG] Raw model output: {res}, predicted class: {actions[pred_class]} ({res[pred_class]*100:.1f}%)")

                # Checking consistency of predictions greater than threshold
                if len(predictions) >= 10 and \
                   np.unique(predictions[-10:])[0] == pred_class and \
                   res[pred_class] > THRESHOLD:

                    action = actions[pred_class]
                    conf = res[pred_class] * 100
                    print(f"[INFO] Consistent prediction: {action} @ {conf:.1f}%")

                    # Updating sentence
                    if not sentence or action != sentence[-1]:
                        sentence = [action]
                        accuracy = [f"{conf:.1f}%"]

        else:
            # No hand: clear everything
            print("[DEBUG] No hand detected — clearing buffers")
            sequence.clear()
            predictions.clear()
            sentence.clear()
            accuracy.clear()

        # Show ROI
        cv2.imshow('ROI', image)

        # Overlay output
        cv2.rectangle(frame, (0, 0), (400, 40), (245, 117, 16), -1)
        out_text = sentence[-1] + " " + accuracy[-1] if sentence else "—"
        cv2.putText(frame, f"Output: {out_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed — exiting.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Cleanup done, program terminated.")





