import cv2
import mediapipe as mp
import time
import numpy as np
import pyrealsense2 as rs

CAMERA_SERIAL = "025222070771"
WIDTH, HEIGHT, FPS = 640, 480, 30

# start camera
cfg = rs.config()
cfg.enable_device(CAMERA_SERIAL)
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

pipe = rs.pipeline()

try:
    profile = pipe.start(cfg)
except RuntimeError as e:
    if not pipe:
        print("Camera not detected. Check the serial number.")

align = rs.align(rs.stream.color)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:

    frameset = pipe.wait_for_frames()
    frameset = align.process(frameset)

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if id == 0 :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                if id == 9 :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                if id == 17 :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                # if id == 4 :
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                # if id == 8 :
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()