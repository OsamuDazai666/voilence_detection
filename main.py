import cv2
from ultralytics import YOLO
from itertools import count
from helper.functions import get_midpoint, cluster_points
from math import dist
import numpy as np

model = YOLO("models/yolov8s-pose.pt")
vid = cv2.VideoCapture("videos/fight2.mov")

cv2.namedWindow("results", cv2.WINDOW_NORMAL) # for resizable window

width = vid.get(3)
height = vid.get(4)

frame_counter = count(start=0, step=1)
frame_count = 0

right_wrist, left_wrist = [10, 9]
right_elbow, left_elbow = [8, 7]
right_ankle, left_ankle = [16, 15]

while True:
    ret, frame = vid.read()

    if not ret:
        print("Video Ended...")
        break 

    if frame_count % 8 == 0:
        result = model.track(frame, verbose=False, conf=0.03, iou=0.5)[0]
        bboxes = result.boxes.xyxy.cpu().numpy().astype("uint32")
        keypoints = result.keypoints.xy.cpu().numpy().astype("uint32")

        mid_points = [get_midpoint(bbox) for bbox in bboxes]
        cluster, index = cluster_points(mid_points, 500)

        filtered_index = []
        for c_point, i in zip(cluster, index):
            if len(c_point) > 1:
                frame = cv2.polylines(img=frame, pts=np.int32([c_point]), isClosed=False, color=(0, 0, 255), thickness=5)
                filtered_index.append(i)


        potentail_voilence_kp = []
        potentail_voilence_zone = []
        for idx_list in filtered_index:
            temp = []
            temp_kp = []
            for i in idx_list:
                temp.append(bboxes[i])
                extracted_keypoints = [
                    keypoints[i][right_ankle],
                    keypoints[i][left_ankle], 
                    keypoints[i][right_elbow],
                    keypoints[i][left_elbow],
                    keypoints[i][right_wrist],
                    keypoints[i][left_wrist],
                ]
                temp_kp.append(extracted_keypoints)

                for point in extracted_keypoints:
                    if point[0] == 0 and point[1] == 0:
                        continue
                    frame = cv2.circle(frame, center=point, radius=12, color=(0, 255, 0), thickness=-1)
            
            potentail_voilence_kp.append(temp_kp)

            t = np.int32(temp)
            x1 = np.min(t[:, 0]) -20
            y1 = np.min(t[:, 1]) -20
            x2 = np.max(t[:, 2]) +20
            y2 = np.max(t[:, 3]) +20

            potentail_voilence_zone.append([x1, y1, x2, y2])

            # draw the red rectangle highlighting potential volience area
            frame = cv2.putText(
                frame, 
                "Potential Voilence Area!!!", 
                (x1 + abs(x1-x2)//2, abs(y1-20)), 
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                fontScale=1,  
                color=(0, 0, 255), 
                thickness=3
            )
            frame = cv2.rectangle(frame, pt1=np.int32([x1, y1]), pt2=np.int32([x2, y2]), color=(0, 0, 255), thickness=3)

        # check the potential areas for actual voilence
        distances = []
        for i, key_pts in enumerate(potentail_voilence_kp):
            distance = []
            for j, kp in enumerate(key_pts):
                r_a, l_a, r_e, l_e, r_w, l_w = kp
                for k in range(0, len(key_pts)):
                    if j == k:
                        continue
                    r_a1, l_a1, r_e1, l_e1, r_w1, l_w1 = key_pts[k]

                    dist_ra = dist(r_a, r_a1)
                    dist_la = dist(l_a, l_a1)
                    dist_re = dist(r_e, r_e1)
                    dist_le = dist(l_e, l_e1)
                    dist_rw = dist(r_w, r_w1)
                    dist_lw = dist(l_w, l_w1)

                    distance.append([
                        dist_ra,
                        dist_la,
                        dist_re,
                        dist_le,
                        dist_rw,
                        dist_lw,
                    ])

            distances.append(distance)


        for distance, pts in zip(distances, potentail_voilence_zone):
            distance = np.int32(distance)
            x1, y1, x2, y2 = pts
            r_a_min, l_a_min, r_e_min = np.min(distance[:, 0]), np.min(distance[:, 1]), np.min(distance[:, 2])
            l_e_min, r_w_min, l_w_min = np.min(distance[:, 3]), np.min(distance[:, 4]), np.min(distance[:, 5])

            if (
                (r_a_min < 180 or l_a_min < 180) and
                (r_e_min < 180 or l_e_min < 180)  
            ):
                if len(distance) <= 2:
                    frame = cv2.putText(
                        frame, 
                        "Intensity Low", 
                        (x1 + abs(x1-x2)//2, abs(y2+20)), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                        fontScale=2,  
                        color=(255, 255, 0), 
                        thickness=3
                    )
                else:
                    frame = cv2.putText(
                        frame, 
                        "Intensity High", 
                        (x1 + abs(x1-x2)//2, abs(y2+20)), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                        fontScale=2,  
                        color=(255, 255, 0), 
                        thickness=3
                    )
            
        # frame = result.plot()
        cv2.imshow("results", cv2.resize(frame, (int(width//2), int(height//2))))

    frame_count = next(frame_counter)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
vid.release()
