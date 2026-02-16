import cv2
from ultralytics import YOLO
import numpy as np
import math

# Load the YOLO26n model
model = YOLO('yolo26n.pt')


# Open the video file
#video_path = "coffeecup.mp4"

#Or open web cam stream
video_path = 0
cap = cv2.VideoCapture(video_path)


# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO26 inference on the frame
        results = model(frame, classes=[67])

        for result in results:
            widest = 0
            boxes = result.boxes.xyxy.numpy()
            closest = [0, 0, 0, 0]
            
            for box in boxes:
                width = abs(box[0] - box[2])
                if width > widest:
                    closest = box
                    widest = width

           
            rwidth = widest/frame.shape[1]
            B = int(rwidth * 255)
    
            frame = cv2.rectangle(frame, (int(closest[0]), int(closest[1])), (int(closest[2]), int(closest[3])), (0, 0, B), 20)

              
        # Display the annotated frame
        frame = cv2.resize(frame, None, fx=1.3, fy=1.3, interpolation = cv2.INTER_NEAREST)
        cv2.imshow("YOLO26 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()