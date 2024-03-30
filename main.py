import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# Define a vertical line splitting the screen into two halves
line_x = 510
line_start = (line_x, 0)
line_end = (line_x, 500)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Check available camera indices
for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Camera index {i} not available")
    else:
        print(f"Camera index {i} available")
        cap.release()
        break  # Use the first available camera index

# Use the first available camera index
cap = cv2.VideoCapture(i)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count_left = 0
count_right = 0
cooldown_time = 30  # Cooldown time in frames
cooldown_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    person_detected = False

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
            # Check if the person is on the left or right side of the line
            if x1 < line_x and x2 < line_x:
                if not person_detected:  # Check if person was not detected in previous frames
                    count_left += 1
                    person_detected = True
            elif x1 > line_x and x2 > line_x:
                if not person_detected:  # Check if person was not detected in previous frames
                    count_right += 1
                    person_detected = True

    # Apply cooldown
    if person_detected:
        cooldown_counter = cooldown_time
    elif cooldown_counter > 0:
        cooldown_counter -= 1

    cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
    cv2.putText(frame, str('Left Count: {}'.format(count_left)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 255), 1)
    cv2.putText(frame, str('Right Count: {}'.format(count_right)), (800, 50), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 255), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
