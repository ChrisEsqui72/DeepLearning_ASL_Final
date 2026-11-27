from pathlib import Path

import random
import torch
import os
import pygame
import time
import cv2 as cv
import torch.nn as nn
import numpy as np
import mediapipe as mp
import torchaudio as ta
# from chatterbox.tts import ChatterboxTTS
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from rembg import remove

# ---- Settings you may want to change (all in one place) ----
SEED       = 42
IMG_SIZE   = 128      # images are resized to IMG_SIZE x IMG_SIZE. Lower = faster on CPU (try 48 or 32 if slow)
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
DATA_DIR   = "./asl_dataset/"
VALID_LABELS = set("0123456789abcdefghijklmnopqrstuvwxyz")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
BEST_PATH = "asl_cnn_best.pt"
WINDOWS_IP = "172.24.0.1" 
STREAM_URL = f"http://{WINDOWS_IP}:5000/video_feed"

random.seed(SEED); torch.manual_seed(SEED)
pygame.mixer.init()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Training on:", device)

class ASLNet(nn.Module):
    def __init__(self, n_classes=36):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

paths, labels = [], []
for p in sorted(Path(DATA_DIR).rglob("*")):
    if p.suffix.lower() in IMG_EXTS and p.parent.name.lower() in VALID_LABELS:
        paths.append(p)
        labels.append(p.parent.name.lower())

classes = sorted(set(labels))

mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def load_models():
    model = ASLNet(len(classes)).to(device)
    # voice_model = ChatterboxTTS.from_pretrained(device=device)

    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()

    return model, None

class ASLTOSPEECH():
    def __init__(self, model, voice_model):
        self.model = model
        self.voice_model = voice_model

    def process_image(self, frame, *, detector=None, input_is_rgb=False):
        detector = hands if detector is None else detector
        h, w = frame.shape[:2]

        image_rgb = frame if input_is_rgb else cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = detector.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        landmarks = results.multi_hand_landmarks[0].landmark
        points = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks],
            dtype=np.float32,
        )
        return cv.boundingRect(points.astype(np.int32)), points

    def get_char(self, hand):
        if hand.size == 0:
            return []

        h, w = hand.shape[:2]
        side = max(h, w)

        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left

        hand = cv.copyMakeBorder(
            hand, top, bottom, left, right,
            cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        hand = cv.cvtColor(hand, cv.COLOR_BGR2RGB)
        hand = cv.resize(hand, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_LINEAR)

        img = torch.from_numpy(hand.astype(np.float32) / 255.0)
        img = img.permute(2, 0, 1).to(device)

        mean = torch.tensor([0.22287842, 0.15386021, 0.13853467], device=device).view(3, 1, 1)
        std = torch.tensor([0.29770571, 0.21619110, 0.20494403], device=device).view(3, 1, 1)
        img = ((img - mean) / std).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(self.model(img), dim=1)[0]

        top = probs.topk(3)
        return [(classes[i], round(p.item(), 3)) for p, i in zip(top.values, top.indices)], hand

    def predict_from_crop(self, crop):
        return self.get_char(remove(crop))
    
    def test(self):
        test_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

        try:
            for path in ["a.jpeg", "b.jpeg", "c.jpeg", "d.jpeg", "e.jpeg", "g.jpeg"]:
                img = np.array(Image.open(path).convert("RGB"))
                box, points = self.process_image(
                    img, detector=test_hands, input_is_rgb=True
                )

                if box is None:
                    print(f"No hand detected in {path}")
                    continue

                print(path, box)
                x, y, box_w, box_h = box
                x1 = max(0, x - 75)
                y1 = max(0, y - 75)
                x2 = min(img.shape[1], x + box_w + 75)
                y2 = min(img.shape[0], y + box_h + 75)
                crop = img[y1:y2, x1:x2].copy()

                cv.imshow("crop", cv.cvtColor(crop, cv.COLOR_RGB2BGR))
    
                preds, hand = self.predict_from_crop(crop)
                print(preds)
                cv.imshow("hand", hand)
                key = cv.waitKey(0) & 0xFF
        finally:
            test_hands.close()

    def start(self):
        cap = cv.VideoCapture(STREAM_URL)
        padding = 25

        if not cap.isOpened():
            raise RuntimeError("Could not open the camera stream.")

        inference_pool = ThreadPoolExecutor(max_workers=1)
        inference_future = None

        sample_interval = 0.10       # Sample at most about 10 times per second
        settle_seconds = 0.35         # Require this much steady time
        max_motion_fraction = 0.02    # Landmark movement relative to hand size
        min_sharpness = 90.0          # Tune for your camera and image size

        next_sample_at = 0.0
        previous_points = None
        stable_since = None
        best_crop = None
        best_sharpness = float("-inf")
        captured_pose = False
        text = ""

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                if inference_future is not None and inference_future.done():
                    predictions, hand = inference_future.result()
                    cv.imshow("isolated_hand", hand)
                    inference_future = None
                    if predictions:
                        print(predictions)
                        text += predictions[0][0]
                        print(text)

                now = time.monotonic()

                # Set the next sample time after processing completes. This avoids
                # immediately processing another frame if hand detection was slow.
                if now >= next_sample_at:
                    box, points = self.process_image(frame)
                    sampled_at = time.monotonic()
                    next_sample_at = sampled_at + sample_interval

                    if points is None:
                        previous_points = None
                        stable_since = None
                        best_crop = None
                        best_sharpness = float("-inf")
                        captured_pose = False
                    else:
                        x, y, box_w, box_h = box
                        hand_size = max(box_w, box_h, 1)

                        if previous_points is None:
                            stable_since = sampled_at
                            best_crop = None
                            best_sharpness = float("-inf")
                            captured_pose = False
                        else:
                            movement = np.linalg.norm(
                                points - previous_points, axis=1
                            ).mean() / hand_size

                            if movement > max_motion_fraction:
                                stable_since = sampled_at
                                best_crop = None
                                best_sharpness = float("-inf")
                                captured_pose = False

                        previous_points = points.copy()

                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + box_w + padding)
                        y2 = min(frame.shape[0], y + box_h + padding)
                        crop = frame[y1:y2, x1:x2].copy()

                        if crop.size > 0 and not captured_pose:
                            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                            sharpness = cv.Laplacian(
                                gray, cv.CV_64F
                            ).var()

                            if sharpness > best_sharpness:
                                best_sharpness = sharpness
                                best_crop = crop.copy()

                            settled = (
                                stable_since is not None
                                and sampled_at - stable_since >= settle_seconds
                            )
                            if (
                                settled
                                and best_sharpness >= min_sharpness
                                and inference_future is None
                            ):
                                print("captured")
                                inference_future = inference_pool.submit(
                                    self.predict_from_crop, best_crop.copy()
                                )
                                captured_pose = True

                        cv.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            2,
                        )

                cv.imshow("Camera", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            inference_pool.shutdown(wait=True)
            cv.destroyAllWindows()

def main():
    model, voice_model = load_models()
    asl_module = ASLTOSPEECH(model, voice_model)

    asl_module.test()

if __name__ == "__main__":
    main()