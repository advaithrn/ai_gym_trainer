import numpy as np
import mediapipe as mp
import os
import cv2
import time

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

lstate = 0
lcounter = 0
rstate = 0
rcounter = 0

def calc_angle(a,b,c):
    radians=np.arctan2(c[1]-b[1], c[0]-b[0])-np.arctan2(a[1]-b[1], a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    if angle>180:
       angle=360-angle;
    return angle

def scrn_print(lcounter,rcounter):
    os.system("cls")
    print("Left Biceps", lcounter, "  Right Biceps", rcounter)
x=time.time()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Pose is an estimation model
    while cap.isOpened():
        # CV2 LIBRARIE'S imread() function reads images in BGR format. But pillow or other models reads in RGB format.
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Re-ordeqring the color frames
        image.flags.writeable = False  # Setting the image to read only saves memory

        # MAKE DETECTION
        results = pose.process(image)  # FOR IMAGE DETECTION
        # from detection we get landmarks not connections
        # connections are later made using draw_landmarks()

        # RECOLOR IMAGE AS PER CV2
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        y=time.time()
        #A DELAY OF 5 SECONDS INDUCED
        if y-x > 5:
            try:
                landmarks = results.pose_landmarks.landmark
                la = np.array((landmarks[11].x, landmarks[11].y), dtype='float')
                lb = np.array((landmarks[13].x, landmarks[13].y), dtype='float')
                lc = np.array((landmarks[15].x, landmarks[15].y), dtype='float')
                langle = calc_angle(la, lb, lc)
                ra = np.array((landmarks[12].x, landmarks[12].y), dtype='float')
                rb = np.array((landmarks[14].x, landmarks[14].y), dtype='float')
                rc = np.array((landmarks[16].x, landmarks[16].y), dtype='float')
                rangle = calc_angle(ra, rb, rc)
                cv2.putText(image, str(lcounter), tuple(np.multiply(lb, [1280, 720]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(rcounter), tuple(np.multiply(rb, [1280, 720]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                if langle > 160:
                    lstate = 0
                elif langle < 30 and lstate == 0:
                    lstate = 1
                    lcounter += 1
                    scrn_print(lcounter,rcounter)

                if rangle > 160:
                    rstate = 0
                elif rangle < 30 and rstate == 0:
                    rstate = 1
                    rcounter += 1
                    scrn_print(lcounter,rcounter)
            except:
                pass
        # TO RENDER DETECTION, FOR AN IMAGE ALL MEDIAPIPE CONNECTIONS ARE DRAWN AND DISPLAYED

        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()