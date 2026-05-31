import cv2
import mediapipe as mp
import time
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from collections import deque
from queue import Queue
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from pathlib import Path
import shutil
import filtering

XARM_IP = "192.168.1.222"
HAND_CAMERA_SERIAL = "025222070771"
ROBOT_CAMERA_SERIAL = "243522071742"
WIDTH, HEIGHT, FPS = 640, 480, 30
REPO_NAME = "clara/testing"
TASK_DESCRIPTION = "Move the glove to the lower lefthand corner of the workspace."

# Hard-coded robot eef position constraints (mm)
robot_x_bounds = [485,180]
robot_y_bounds = [345,-210]
robot_z_bounds = [400,172]
boundary = robot_x_bounds + robot_y_bounds + robot_z_bounds

pinch_threshold_close = 0.04
pinch_threshold_open = 0.08

# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("R.txt")
t = np.loadtxt("t.txt")

# ---------------- GLOBAL STATE ----------------
arm = XArmAPI(XARM_IP)
uv = None
pose = None
distance = []
obs_queue = Queue()
recording = False
controlling = True

# start cameras
def make_pipeline(serial):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
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

def get_image(pipe,align,depth_bool=False):
    try:
        frameset = pipe.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        print("Frame timeout, skipping...")
        return None,None

    if depth_bool:
        frameset = align.process(frameset)
        depth_frame = frameset.get_depth_frame()

    else: depth_frame = None

    color_frame = frameset.get_color_frame()

    img = np.asanyarray(color_frame.get_data())

    return img, depth_frame

pipe_hand, cfg_hand, align_hand = make_pipeline(HAND_CAMERA_SERIAL)
time.sleep(2)
pipe_robot, cfg_robot, align_robot = make_pipeline(ROBOT_CAMERA_SERIAL)


# Initialize mediapipe handtracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1
        )
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Initialize dataset
dataset_path = HF_LEROBOT_HOME / REPO_NAME

if Path(dataset_path/"meta").exists() and Path(dataset_path/"data").exists():
    dataset = LeRobotDataset(
        root=dataset_path,
        repo_id=REPO_NAME,
    )
    print("Adding to existing dataset, waiting for signal.")
else:
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=FPS,
        features={
            "robot_camera": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "hand_camera": { # this one is not used, put it as zeros or something
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "extra_camera": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "eef_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                "names": ["actions"],
            },
        },
    )

# Start up robot
"""Initialize XArm for control with safety limits."""
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)

arm.motion_enable(enable=True)
arm.set_mode(7)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

# Hardware-level safety boundaries (mm)
# Format: [x_max, x_min, y_max, y_min, z_max, z_min]
# print(boundary)
# arm.set_reduced_tcp_boundary(boundary)
# arm.set_fense_mode(True)
arm.set_reduced_mode(False)

# Conservative acceleration limits for teleoperation
arm.set_tcp_maxacc(5000)            # mm/s^2
arm.set_joint_maxacc(10)            # rad/s^2
arm.set_reduced_max_tcp_speed(200)  # mm/s
arm.set_reduced_max_joint_speed(60) # deg/s
arm.set_reduced_mode(True)

print('XArm initialized with safety limits')

landmark_filter = filtering.LowPassFilter(alpha=0.1)
recording = False
video_writer = None

while True:
    code, state = arm.get_state()
    if state == 4 or state == 5:
        print("Cleaning error")
        arm.clean_error()

        arm.motion_enable(enable=True)
        arm.set_mode(7)
        arm.set_state(0)
        arm.set_gripper_enable(enable=True)
        arm.set_gripper_mode(0)

    hand_img, depth_frame = get_image(pipe_hand, align_hand, True)
    robot_img,_ = get_image(pipe_robot,align_robot)

    if hand_img is None or robot_img is None:
        continue

    imgRGB = hand_img.copy()
    results = hands.process(imgRGB)

    thumb_3d = None
    index_3d = None

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
        for handLms, worldLms in zip(results.multi_hand_landmarks,
                                    results.multi_hand_world_landmarks):

            # --- pixel coords (for visualization / depth lookup) ---
            h, w, _ = hand_img.shape

            def to_pixel(lm):
                cx = int(np.clip(lm.x * w, 0, w - 1))
                cy = int(np.clip(lm.y * h, 0, h - 1))
                return [cx, cy]

            uv = to_pixel(handLms.landmark[5])  # knuckle (pixel)
            

            # --- world coordinates (meters) ---
            thumb_w = worldLms.landmark[4]
            index_w = worldLms.landmark[8]

            thumb_3d = np.array([thumb_w.x, thumb_w.y, thumb_w.z])
            index_3d = np.array([index_w.x, index_w.y, index_w.z])

            # --- draw (still uses pixel coords) ---
            # cv2.circle(hand_img, tuple(to_pixel(handLms.landmark[4])), 10, (255,0,255), -1)
            # cv2.circle(hand_img, tuple(to_pixel(handLms.landmark[8])), 10, (255,0,255), -1)

            # mpDraw.draw_landmarks(hand_img, handLms, mpHands.HAND_CONNECTIONS)

    if thumb_3d is not None and index_3d is not None:
        # Calculate distance (meters)
        d = np.linalg.norm(thumb_3d - index_3d)
        if d <= 0.06:
            grasp = 0
        else:
            grasp = 850

        code, grip_curr = arm.get_gripper_position()
        # print(grip_curr)

        if grip_curr < 0:
            if grasp == 0:
                grip_cmd = 0
            else:
                grip_cmd = np.min([850, grip_curr+100])
        elif grasp < grip_curr:
            grip_cmd = np.max([0, grip_curr-100])
        else:
            grip_cmd = np.min([850, grip_curr+100])

        if controlling:
            arm.set_gripper_position(int(grip_cmd), wait=False)
    
    # if thumb_3d is not None and index_3d is not None:
    #     # Calculate distance (meters)
    #     d = np.linalg.norm(thumb_3d - index_3d)
    #     distance.append(d)
    #     # print(f'distance: {d}')

    #     if len(distance) == 3:
    #         avg_d = np.mean(np.array(distance))
    #         # Apply hysteresis for stability
    #         if avg_d <= pinch_threshold_close:
    #             gripper = 1.0  # Closed
    #         elif avg_d >= pinch_threshold_open:
    #             gripper = 0.0  # Open
    #         else:
    #             # Linear interpolation in the middle zone
    #             range_size = pinch_threshold_open - pinch_threshold_close
    #             gripper = 1.0 - (avg_d - pinch_threshold_close) / range_size

                
    #         grasp = 850 - 850 * gripper
            
    #         if controlling:
    #             arm.set_gripper_position(int(grasp), wait=False)

    #         distance = []
            

    if uv is not None:
        depth = depth_frame.get_distance(uv[0],uv[1])
        if depth < 0 or depth > 2:
            print("Invalid depth")
        else:
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            xyz = landmark_filter.update(rs.rs2_deproject_pixel_to_point(depth_intrin, uv, depth))
            pose = R @ np.array(xyz)*1000 + t
            # cv2.putText(hand_img, str(np.round(xyz,2)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
            #     (255, 0, 255), 3)

    if pose is not None:
        # Software boundary constraint to avoid triggering a controller error
        for i in range(len(pose)):
            pose_max = boundary[2*i]
            pose_min = boundary[2*i+1]
            if pose[i] > pose_max:
                pose[i] = pose_max - 3
            if pose[i] < pose_min:
                pose[i] = pose_min + 3

        if controlling:
            # arm.set_position(x=pose[0], y=pose[1], z=pose[2],
            #                 roll=-175, pitch=-5, yaw=-5,
            #                 is_radian=False, wait=False
            #                 )
            arm.set_position(x=pose[0], y=pose[1], z=pose[2],
                roll=-180, pitch=0, yaw=0,
                is_radian=False, wait=False
                )

    # cv2.putText(hand_img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #             (255, 0, 255), 3)

    eef_pose = arm.get_position()[1]

    # Convert [-180,180] to [0,360] degrees
    eef_pose[3] = eef_pose[3] % 360
    eef_pose[5] = eef_pose[5] % 360

    # Convert roll, pitch, yaw from degrees to radians
    angles_rad = (np.array(eef_pose[3:6]) * np.pi / 180).tolist()
    eef_state = np.array(eef_pose[:3] + angles_rad, dtype=np.float32)

    _, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860, dtype=np.float32)

    total_state = np.concatenate((eef_state,np.array([g_p],dtype=np.float32)))
    
    observation = {
            "eef_position": eef_state,
            "gripper_position": g_p.reshape(1,),
            "actions": total_state,
            "robot_camera": np.asanyarray(cv2.resize(robot_img, (320, 240)).copy()),
            "hand_camera": np.asanyarray(cv2.resize(hand_img, (320, 240)).copy()),
            "extra_camera": np.zeros_like(cv2.resize(hand_img, (320, 240))),
            "task": TASK_DESCRIPTION,
        }
    
    if recording:
        obs_queue.put(observation)
        if video_writer is not None:
            video_writer.write(combined_frame)

    hand_imgBGR = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    robot_imgBGR = cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB)
    
    cv2.imshow("Robot Image", robot_imgBGR)
    cv2.imshow("Hand Image", hand_imgBGR)

    combined_frame = np.hstack([
        hand_imgBGR,
        robot_imgBGR
    ])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    print(fps)
    pTime = cTime
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('x'):
        print('Stopping demo')
        print(f'Episode length -- {obs_queue.qsize()} frames')
        controlling = False
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None

        print("Recording stopped")

    if key == ord('g'):
        print('Starting control')
        controlling = True

    if key == ord('r'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            f"teleop_{int(time.time())}.mp4",
            fourcc,
            12,
            (WIDTH * 2, HEIGHT)
        )
        print('Recording demo')

        recording = True

    if key == ord('s'):
        print('Saving demo')
        prev_obs = None
        frames_recorded = 0
        while not obs_queue.empty():
            
            obs = obs_queue.get()

            if prev_obs is not None:
                prev_obs["actions"] = obs["actions"]
                dataset.add_frame(prev_obs)
                frames_recorded += 1

            prev_obs = obs
        dataset.save_episode()
        recording = False
        print(f'Demo saved with {frames_recorded} frames.')

    if key == ord('d'):
        print('Deleting demo')
        obs_queue = Queue()
        recording = False

    if key == ord('q'):
        break

cv2.destroyAllWindows()


