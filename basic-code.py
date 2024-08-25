import cv2
import torch
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture(0)

# Loop to continuously get frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    results = model(frame)
    
    # Draw the bounding boxes and labels on the frame
    annotated_frame = results[0].plot()
    
    # Display the resulting frame
    cv2.imshow('YOLOv5 Live Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
