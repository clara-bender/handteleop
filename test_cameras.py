import pyrealsense2 as rs
import cv2

HAND_CAMERA_SERIAL = "025222070771"
ROBOT_CAMERA_SERIAL = "243522071742"

WIDTH, HEIGHT, FPS = 320, 240, 30

def make_pipeline(serial):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipe = rs.pipeline()
    align = rs.align(rs.stream.color)

    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"\nFAILED START for {serial}")
        print("ERROR:", e)
        return None, None, None
    return pipe, cfg, align

pipe_hand, cfg_hand, align_hand = make_pipeline(HAND_CAMERA_SERIAL)

pipe_robot, cfg_robot, align_robot = make_pipeline(ROBOT_CAMERA_SERIAL)


while True:
    try:
        frames = pipe_hand.wait_for_frames(1000)
        cv2.show(frames)
        print("Hand frame ok")
    except:
        print("Hand frame NOT ok")

    try:
        frames = pipe_robot.wait_for_frames(1000)
        print("Robot frame ok")
    except:
        print("Robot frame NOT ok")