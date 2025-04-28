import cv2
import torch
import numpy as np
from ultralytics import YOLO

# DeepSORT dependencies: You might need to install these
# !pip install filterpy
# !pip install lap
import sys
sys.path.insert(0, 'deep_sort_pytorch')  # Replace with the actual path
from deep_sort_pytorch.utils.parser import get_config

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# 1. Configuration
VIDEO_PATH = 'hippo2.mp4'  # Replace with your video path
YOLO_MODEL_PATH = 'model.pt'  # Replace with your YOLOv8 model path
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO detections
IOU_THRESHOLD = 0.5  # IOU threshold for NMS in YOLO
MAX_IOU_DISTANCE = 0.7  # Max IoU distance for DeepSORT matching
NN_BUDGET = 100  # Max number of appearance descriptors to keep per track

# 2. Initialize YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)
class_names = model.names

# 3. Initialize DeepSORT tracker
def initialize_deepsort():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,
                        n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=NN_BUDGET,
                        use_cuda=torch.cuda.is_available())
    return deepsort

deepsort_tracker = initialize_deepsort()

# 4. Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

# 5. Main Processing Loop
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    img = cv2.resize(frame, (1020, 500))

    # 6. Object Detection with YOLOv8
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)  # Apply confidence threshold

    # Extract detections (boxes, confidence, class IDs)
    #detections = []
    bboxes_xywh = []
    confidence_scores = []
    class_ids = []


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())

            '''detections.append([x1, y1, x2 - x1, y2 - y1])  # DeepSORT expects x, y, width, height
            confidence_scores.append(confidence)
            class_ids.append(class_id)'''

            # bounding box with xywh format
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2
            bboxes_xywh.append([x_center, y_center, width, height])
            confidence_scores.append(confidence)
            class_ids.append(class_id)

    #convert data to np arrays
    bboxes_xywh = np.array(bboxes_xywh,dtype=np.float32)
    confidence_scores = np.array(confidence_scores,dtype=np.float32)
    class_ids = np.array(class_ids)  # Ensure class_ids is a numpy array
    # Only select detections with confidence above the threshold

    '''mask = confidence_scores>CONFIDENCE_THRESHOLD
    bboxes_xywh=bboxes_xywh[mask]
    confidence_scores = confidence_scores[mask]
    class_ids = class_ids[mask]'''

    # 7. DeepSORT Tracking
    if len(bboxes_xywh)>0:
        # Reshape image to ensure it has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # lets add a check for image shape
            print(f"Image shape: {img.shape}")
            if len(img.shape) != 3 or img.shape[2] != 3:
                print("ERROR: Image does not have the expected shape (H, W, 3). Exiting.")
                exit()


        #Run DeepSORT tracker
        tracks, _ = deepsort_tracker.update(bboxes_xywh, confidence_scores,img,class_ids)
        #Draw bounding boxes and labels
        for track in tracks:
            if not  track.is_confirmed() or track.time_since_update >1:
                continue   #skip all unconfirmed tracks

            '''bbox = track.to_tlbr().astype(int) #GET bounding box in top-left bottom-right format
            track_id = track.track_id
            class_id = class_ids[0] #Assuming all detections are the same class will need improve later
            '''
            x1,u1,x2,y2,classid,trackid=track
            bbox = [x1,y1,x2,y2]

            cv2.rectangle(img, (bbox[0], bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            cv2.putText(img,f"{class_names[class_id]} ID: {track_id}",(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    '''# Convert detections to NumPy arrays
    detections = np.array(detections)
    confidence_scores = np.array(confidence_scores)

    # 7. DeepSORT Tracking
    if len(detections) > 0:
        # Run DeepSORT tracker
        tracks = deepsort_tracker.update(detections, confidence_scores, img,img)

        # Draw bounding boxes and labels
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue  # Skip unconfirmed tracks

            bbox = track.to_tlbr().astype(int)  # Get bounding box in top-left bottom-right format
            track_id = track.track_id
            class_id = class_ids[0]  # Assuming all detections are the same class - IMPROVE THIS LATER

            # Draw bounding box and label
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{class_names[class_id]} ID: {track_id}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

'''
    # 8. Display Results
    cv2.imshow('YOLOv8 + DeepSORT Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Clean Up
cap.release()
cv2.destroyAllWindows()
