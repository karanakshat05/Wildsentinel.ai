import os
from ultralytics import YOLO
import cv2

# Set directories and file paths
#VIDEOS_DIR = ''
#video_path = ('hippo2.mp4')#
#video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0])

# Check if video file exists
'''if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
    '''

# Initialize video capture and output
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

'''if not ret:
    raise ValueError(f"Unable to read the video file: {video_path}")
    '''

H, W, _ = frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
'''
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H))
'''

# Load the YOLO model
#model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
#if not os.path.exists(model_path):
#    raise FileNotFoundError(f"Model weights not found: {model_path}")
model_path='model.pt'
model = YOLO(model_path)  # Load the custom YOLO model

threshold = 0.5  # Detection threshold

# Process the video
while ret:
    results = model(frame)[0]  # Perform inference

    for result in results.boxes.data.tolist():  # Loop through detections
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label
            label = results.names[int(class_id)].upper()
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('detection',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #out.write(frame)  # Write the frame to the output file
    ret, frame = cap.read()  # Read the next frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

#print(f"Processed video saved at: {video_path_out}")
