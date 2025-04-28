from ultralytics import YOLO
import cv2
import numpy as np


# Load a model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('hippo2.mp4')

#trackers = cv2.MultiTracker_create()
trackers =[]
count = 0
detected_ids = set()


while True:
    ret, img = cap.read()
    if not ret:
        print("no frame") #debug
        break


    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

#start detecting the class(animal)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

    if masks is not None:
        masks = masks.data.cpu().numpy()

        for seg, box in zip(masks, boxes):
            seg = cv2.resize(seg, (w, h))
            _, thresh =cv2.threshold(seg.astype(np.uint8),0.5,255,cv2.THRESH_BINARY)
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)

                if (x, y, x1, y1) not in detected_ids:
                    detected_ids.add((x, y, x1, y1))
                    count += 1



                #cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=1)
                cv2.rectangle(img,(x,y),(x1+x,y1+y),(0,255,0),1)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(img, (x, y, x1 , y1 ))
                trackers.append(tracker)

    for tracker in trackers:
        success, box = tracker.update(img)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(img, f'Total Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()