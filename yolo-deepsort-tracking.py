import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch




class AnimalTracker:
    def __init__(self, model_path, confidence=0.5, iou=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = confidence
        self.iou_threshold = iou

        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
        )

        # Dictionary to store animal counts
        self.animal_counts = {}

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame)[0]

        # Lists for detections
        detection_boxes = []
        detection_scores = []
        detection_classes = []

        # Process YOLO detections
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.conf_threshold:
                detection_boxes.append([x1, y1, x2, y2])
                detection_scores.append(score)
                detection_classes.append(int(class_id))

                # Print detection info for debugging
                print(f"Detection: Class {self.model.names[int(class_id)]} at ({x1:.2f}, {y1:.2f})")

        # Update trackers if we have detections
        if len(detection_boxes) > 0:
            detection_boxes = np.array(detection_boxes)

            # Get tracking updates
            tracks = self.tracker.update_tracks(detection_boxes, detection_scores, detection_classes, frame=frame)

            # Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue

                # Get track info
                track_id = track.track_id
                class_id = track.get_class()
                class_name = self.model.names[class_id]

                # Update animal count for this class
                if class_name not in self.animal_counts:
                    self.animal_counts[class_name] = 1

                # Create label with class name and count
                label = f"{class_name}+{self.animal_counts[class_name]}"
                self.animal_counts[class_name] += 1

                # Get bounding box
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw label with background
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y1_label = max(y1, label_size[1])

                # Print tracking info for debugging
                print(f"Tracking: {label} at position ({x1}, {y1})")

                # Draw label background
                cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10),
                              (x1 + label_size[0], y1_label + baseline - 10),
                              (0, 255, 0), cv2.FILLED)

                # Draw label text
                cv2.putText(frame, label, (x1, y1_label - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Reset counts for next frame
        self.animal_counts = {}

        return frame


def main():
    # Initialize tracker
    tracker = AnimalTracker(
        model_path='model.pt',  # Replace with your model path
        confidence=0.3,
        iou=0.45
    )

    # Initialize video capture
    video_path='hippo2.mp4'
    cap = cv2.VideoCapture(video_path)  # Use 0 for webcam or provide video path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = tracker.process_frame(frame)

        # Display result
        cv2.imshow('Animal Tracking', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()