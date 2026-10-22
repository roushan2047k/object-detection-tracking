
import cv2
import urllib.request
import os
import math

# Model files
prototxt_url = (
    "https://raw.githubusercontent.com/"
    "chuanqi305/MobileNet-SSD/master/deploy.prototxt"
)
model_url = (
    "https://github.com/chuanqi305/MobileNet-SSD/"
    "raw/master/mobilenet_iter_73000.caffemodel"
)

prototxt = "deploy.prototxt"
weights = "mobilenet_ssd.caffemodel"

# Download model files automatically
for url, filename in [
    (prototxt_url, prototxt),
    (model_url, weights)
]:
    if not os.path.exists(filename):
        print("Downloading", filename)
        urllib.request.urlretrieve(url, filename)

# Object classes supported by MobileNet-SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load pretrained model
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam. Check camera permissions.")
    raise SystemExit

# Simple tracking memory
previous_objects = []
next_id = 1

print("Object detection started. Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    current_objects = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < 0.45:
            continue

        class_id = int(detections[0, 0, i, 1])

        if class_id >= len(CLASSES) or class_id == 0:
            continue

        box = detections[0, 0, i, 3:7] * [
            width, height, width, height
        ]

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Match with an object from the previous frame
        matched_id = None
        best_distance = 80

        for old in previous_objects:
            if old["class"] != CLASSES[class_id]:
                continue

            ox, oy = old["center"]
            distance = math.hypot(center[0] - ox, center[1] - oy)

            if distance < best_distance:
                best_distance = distance
                matched_id = old["id"]

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        current_objects.append({
            "center": center,
            "class": CLASSES[class_id],
            "id": matched_id
        })

        label = (
            f"{CLASSES[class_id]} ID:{matched_id} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

    previous_objects = current_objects

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Program stopped.")
