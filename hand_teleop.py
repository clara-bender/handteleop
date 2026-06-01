import cv2
import mediapipe as mp
import time
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import math
import filtering

XARM_IP = '192.168.1.222'
TOP_CAMERA_SERIAL = "025222070771"
BOTTOM_CAMERA_SERIAL = "243122302276"
ROBOT_CAMERA_SERIAL = "243522071742"
WIDTH, HEIGHT, FPS = 640, 480, 30

# Hard-coded robot eef position constraints (mm)
robot_x_bounds = [485,180]
robot_y_bounds = [345,-210]
robot_z_bounds = [600,160]
boundary = robot_x_bounds + robot_y_bounds + robot_z_bounds

pinch_threshold_close = 0.04
pinch_threshold_open = 0.07

# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("R.txt")
t = np.loadtxt("t.txt")

# ---------------- GLOBAL STATE ----------------
arm = XArmAPI(XARM_IP)
uv_wrist = None
grip_cmd = 850
pose = None
distance = []

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

def to_pixel(lm):
    cx = int(np.clip(lm.x * top_w, 0, top_w - 1))
    cy = int(np.clip(lm.y * top_h, 0, top_h - 1))
    return [cx, cy]

top_pipe, top_cfg, top_align = make_pipeline(TOP_CAMERA_SERIAL)
bottom_pipe, bottom_cfg, bottom_align = make_pipeline(BOTTOM_CAMERA_SERIAL)
robot_pipe, robot_cfg, robot_align = make_pipeline(ROBOT_CAMERA_SERIAL)

def initialize_hands():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1
        )
    mpDraw = mp.solutions.drawing_utils
    return mpHands, hands, mpDraw

# Initialize mediapipe handtracking
top_mpHands, top_hands, top_mpDraw = initialize_hands()
bottom_mpHands, bottom_hands, bottom_mpDraw = initialize_hands()

pTime = 0
cTime = 0

# Start up robot
"""Initialize XArm for control with safety limits."""
code, state = arm.get_state()
if state != 0:
    arm.clean_error()
    time.sleep(0.5)

arm.motion_enable(enable=True)
arm.set_mode(7)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

# Hardware-level safety boundaries (mm)
# Format: [x_max, x_min, y_max, y_min, z_max, z_min]
print(boundary)
arm.set_reduced_tcp_boundary(boundary)
arm.set_fense_mode(True)

# Conservative acceleration limits for teleoperation
arm.set_tcp_maxacc(5000)            # mm/s^2
arm.set_joint_maxacc(10)            # rad/s^2
arm.set_reduced_max_tcp_speed(600)  # 200mm/s
arm.set_reduced_max_joint_speed(60) # deg/s
arm.set_reduced_mode(True)

print('XArm initialized with safety limits')

landmark_filter = filtering.LowPassFilter(alpha=0.2)
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

    top_img, top_depth_frame = get_image(top_pipe, top_align, True)
    bottom_img, _ = get_image(bottom_pipe, bottom_align)
    robot_img, _ = get_image(robot_pipe, robot_align)

    top_imgBGR = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
    bottom_imgBGR = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB)
    robot_imgBGR = cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB)

    top_results = top_hands.process(top_img)
    bottom_results = bottom_hands.process(bottom_img)

    pitch_deg = None
    # --- pixel coords (for visualization / depth lookup) ---
    top_h, top_w, _ = top_img.shape
    bottom_h, bottom_w, _ = bottom_img.shape

    thumb_3d = None
    index_3d = None

    # Extract palm landmarks from top camera view in order to calculate x,y,z,roll,pitch,yaw
    if top_results.multi_hand_landmarks and top_results.multi_hand_world_landmarks:
        for handLms, worldLms in zip(top_results.multi_hand_landmarks,
                                    top_results.multi_hand_world_landmarks):
            
            uv_wrist = to_pixel(handLms.landmark[0])  # wrist (pixel)
            uv_middle_knuckle = to_pixel(handLms.landmark[9])  # ring knuckle (pixel)
            uv_pinky_knuckle = to_pixel(handLms.landmark[17])  # pinky knuckle (pixel)
            uv_index_knuckle = to_pixel(handLms.landmark[9]) # index knuckle (pixel)

            palm_points2d = [uv_wrist, uv_middle_knuckle, uv_pinky_knuckle, uv_index_knuckle]
            palm_points3d = []

            for point2d in palm_points2d:
                u = point2d[0]
                v = point2d[1]
                if 0 <= u < top_w and 0 <= v < top_h:
                    depth = top_depth_frame.get_distance(u, v)
                    if depth > 0:
                        intrin = top_depth_frame.profile.as_video_stream_profile().intrinsics
                        point3d = rs.rs2_deproject_pixel_to_point(intrin, [u, v], depth)
                        palm_points3d.append(np.array(point3d))

            if len(palm_points3d) == 4:
                wrist3d, middle_knuckle3d, pinky_knuckle3d, index_knuckle3d = landmark_filter.update(palm_points3d)

                def normalized(vector):
                    return vector/np.linalg.norm(vector)

                # vectors for calculating pitch, roll, and yaw
                vec1_curr = pinky_knuckle3d - wrist3d
                vec2_curr = middle_knuckle3d - pinky_knuckle3d

                # perpendicular vector, pointing up
                perp_curr = np.cross(vec1_curr,vec2_curr)

                # Nominal values (constants):
                vec1_nom = np.array([-0.03087179,  0.05859347, -0.00200003])
                vec2_nom = np.array([ 0.03746835,  0.01238391, -0.01099998])
                perp_nom = np.cross(vec1_nom, vec2_nom)

                up_nom = normalized(perp_nom) # Robot +z
                forward_nom = normalized(vec1_nom + vec2_nom) # Robot +x
                left_nom = normalized(np.cross(up_nom, forward_nom)) # Robot +y
                forward_nom = normalized(np.cross(left_nom, up_nom))

                # if np.dot(perp_curr, perp_nom) > 0:
                #     perp_curr = -perp_curr

                # Current values
                up_curr = normalized(perp_curr) # Robot +z
                forward_curr = normalized(vec1_curr + vec2_curr) # Robot +x
                left_curr = normalized(np.cross(up_curr, forward_curr)) # Robot +y
                forward_curr = normalized(np.cross(left_curr, up_curr))

                # Calculate rotation between nominal and current values
                R_nom = np.column_stack((forward_nom, left_nom, up_nom))
                R_curr = np.column_stack((forward_curr, left_curr, up_curr))
                R_rel = R_curr @ R_nom.T

                yaw   = math.atan2(R_rel[1,0], R_rel[0,0])
                pitch = math.asin(-R_rel[2,0])
                roll  = math.atan2(R_rel[2,1], R_rel[2,2])

                def normalize_angle(angle_deg):
                    """
                    Convert any angle to range [-180, 180)
                    """
                    angle = (angle_deg + 180) % 360 - 180
                    return angle

                # Convert to degrees if needed
                yaw_deg   = -math.degrees(yaw)
                pitch_deg = math.degrees(roll)
                roll_deg = -normalize_angle(180-math.degrees(pitch))

                # print(f"yaw: {yaw_deg}")
                # print(f"pitch: {pitch_deg}")
                # print(f"roll: {roll_deg}")

                # Get robot xyz pose from wrist point
                xyz = index_knuckle3d
                pose = R @ np.array(xyz)*1000 + t
            else:
                print("Not all points detected")

            # --- draw (still uses pixel coords) ---
            cv2.circle(top_img, tuple(to_pixel(handLms.landmark[4])), 10, (255,0,255), -1)
            cv2.circle(top_img, tuple(to_pixel(handLms.landmark[8])), 10, (255,0,255), -1)

            top_mpDraw.draw_landmarks(top_img, handLms, top_mpHands.HAND_CONNECTIONS)
    
    else:
        print("No hand detection (top)")
    # Extract thumb and index fingertip from bottom frame for gripper position

    if bottom_results.multi_hand_landmarks and bottom_results.multi_hand_world_landmarks:
        for handLms, worldLms in zip(bottom_results.multi_hand_landmarks,
                                    bottom_results.multi_hand_world_landmarks):
            
            # --- world coordinates (meters) ---
            thumb_w = worldLms.landmark[4]
            index_w = worldLms.landmark[8]

            thumb_3d = np.array([thumb_w.x, thumb_w.y, thumb_w.z])
            index_3d = np.array([index_w.x, index_w.y, index_w.z])
    else:
        print("No hand detection (bottom)")


    if thumb_3d is not None and index_3d is not None:
        # Calculate distance (meters)
        d = np.linalg.norm(thumb_3d - index_3d)
        if d <= 0.06:
            grasp = 0
        else:
            grasp = 850
            print(d)

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

        arm.set_gripper_position(int(grip_cmd), wait=False)


    if pose is not None and pitch_deg is not None:
        # Software boundary constraint to avoid triggering a controller error
        for i in range(len(pose)):
            pose_max = boundary[2*i]
            pose_min = boundary[2*i+1]
            if pose[i] > pose_max:
                pose[i] = pose_max - 1
            if pose[i] < pose_min:
                pose[i] = pose_min + 1

        # arm.set_position(x=pose[0], y=pose[1], z=pose[2],
        #                 roll=-175, pitch=-5, yaw=-5,
        #                 is_radian=False, wait=False
        #                 )
        arm.set_position(x=pose[0], y=pose[1], z=pose[2],
                        roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
                        is_radian=False, wait=False
                        )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    print(fps)
    pTime = cTime

    cv2.imshow("Top Image", top_imgBGR)
    cv2.imshow("Bottom Image", bottom_imgBGR)
    cv2.imshow("Robot Image", robot_imgBGR)

    combined_frame = np.hstack([
        top_imgBGR,
        bottom_imgBGR,
        robot_imgBGR
    ])

    if recording and video_writer is not None:
        video_writer.write(combined_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r') and not recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(
            f"teleop_{int(time.time())}.mp4",
            fourcc,
            12,
            (WIDTH * 3, HEIGHT)
        )

        recording = True
        print("Recording started")

    if key == ord('x') and recording:
        recording = False

        if video_writer is not None:
            video_writer.release()
            video_writer = None

        print("Recording stopped")


cv2.destroyAllWindows()


