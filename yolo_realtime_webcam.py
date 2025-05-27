import cv2
import torch

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("📷 Press 'q' in the image window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not read properly")
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("🔍 YOLOv5 Live Detection - Press 'q' to Quit", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("👋 Exiting app...")
        break

cap.release()
cv2.destroyAllWindows()
