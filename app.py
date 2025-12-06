import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("hand_gesture_mobilenett.h5")
labels = ["fist", "ok", "palm", "peace"]

running = False  


def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, 0)
    return img


def predict_stream(frame):
    global running
    if not running:
        return None, "Stopped", {}

    if frame is None:
        return None, "Waiting...", {}

    # ROI area
    h, w, _ = frame.shape
    x1, y1 = w // 2 - 120, h // 2 - 120
    x2, y2 = w // 2 + 120, h // 2 + 120

    roi = frame[y1:y2, x1:x2]

    img = preprocess(roi)
    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    gesture = labels[idx]

    # Annotated frame
    frame_out = frame.copy()
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame_out, f"{gesture} {pred[idx]:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    probs = {labels[i]: float(pred[i]) for i in range(4)}

    return frame_out, gesture, probs


def start():
    global running
    running = True
    return "Detection started!"


def stop():
    global running
    running = False
    return "Detection stopped."


with gr.Blocks() as demo:

    gr.Markdown("## 🤖 Real-Time Hand Gesture Recognition")

    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input")
        output = gr.Image(label="AI Detection Output")

    with gr.Row():
        label_output = gr.Label(label="Detected Gesture")
        probs_output = gr.Label(label="Classification Probabilities")

    with gr.Row():
        start_btn = gr.Button("▶️ Start Detection")
        stop_btn = gr.Button("⏹️ Stop Detection")

    start_btn.click(start, outputs=label_output)
    stop_btn.click(stop, outputs=label_output)

    webcam.stream(
        predict_stream,
        inputs=webcam,
        outputs=[output, label_output, probs_output]
    )

demo.launch(share=True)
