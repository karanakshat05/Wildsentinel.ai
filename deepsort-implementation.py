import os
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path, video_path, confidence=0.5):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = confidence
        self.video_path = video_path

        # Check the file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        print(f"Video opened successfully: {video_path}")
        print(f"Video FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Video resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        # Initialize DeepSORT with default parameters
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )

        # Initialize counter for objects
        self.track_history = {}
        self.class_counter = {}

    def track_objects(self, frame):
        """
        Main tracking function
        """
        # Step 1: Get detections from YOLO
        results = self.model(frame)[0]

        # Step 2: Format detections for DeepSORT
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.conf_threshold:
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                detections.append(([x1, y1, x2, y2], score, int(class_id)))

        # Print detection info
        print(f"Number of detections: {len(detections)}")

        # Step 3: Update DeepSORT tracker
        if detections:
            bboxes = np.array([d[0] for d in detections])
            scores = np.array([d[1] for d in detections], dtype=np.float32)
            class_ids = np.array([d[2] for d in detections], dtype=np.int32)

            print("Shape of bboxes:", bboxes.shape)
            print("Data type of bboxes:", bboxes.dtype)
            print("Contents of bboxes:", bboxes)

            # Pass the frame along with detections to update_tracks
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Step 4: Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue

                # Get track info
                track_id = int(track.track_id)  # Ensure track_id is int
                class_id = track.det_class      # Correct attribute for class id
                bbox = track.to_tlbr()          # Get bounding box coordinates

                # Get class name
                class_name = self.model.names[class_id]

                # Update class counter
                if class_name not in self.class_counter:
                    self.class_counter[class_name] = 1

                # Create unique identifier for this object
                if track_id not in self.track_history:
                    self.track_history[track_id] = f"{class_name}{self.class_counter[class_name]}"
                    self.class_counter[class_name] += 1

                # Draw bounding box and label
                self._draw_track(frame, bbox, self.track_history[track_id], track_id)

                # Print tracking info
                print(f"Tracking {self.track_history[track_id]} at position {bbox}")

        return frame

    def _draw_track(self, frame, bbox, label, track_id):
        """
        Draw bounding box and label for a track
        """
        x1, y1, x2, y2 = map(int, bbox)
        track_id = int(track_id)
        color = self._get_color(track_id)
        # Debug: print color
        # print(f"Color: {color}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    def _get_color(self, idx):
        """
        Generate unique color for each track ID as a tuple of ints
        """
        hue = (idx * 30) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 250, 250]]]), cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, color))  # Ensure tuple of ints

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.track_objects(frame)

            cv2.imshow('Object Tracking', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Define paths
    model_path = os.path.join(current_dir, 'pythonProject/model.pt')
    video_path = os.path.join(current_dir, 'pythonProject/hippo2.mp4')

    print(f"Full model path: {model_path}")
    print(f"Full video path: {video_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Video exists: {os.path.exists(video_path)}")

    try:
        tracker = ObjectTracker(
            model_path=model_path,
            video_path=video_path,
            confidence=0.3
        )
        print("Tracker initialized successfully")
        tracker.process_video()

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
