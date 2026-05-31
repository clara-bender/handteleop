#!/usr/bin/env python3
"""Multi-RealSense viewer with click-to-record XYZ + robot pose pairing."""

import pyrealsense2 as rs
import numpy as np
import cv2
from xarm.wrapper import XArmAPI
import time

# ---------------- CONFIG ----------------
CAMERA_SERIAL = "025222070771"

XARM_IP = '192.168.1.222'
WIDTH, HEIGHT, FPS = 640, 480, 30
MAX_DEPTH_METERS = 2.0

SEND_BUTTON = {"x1": 10, "y1": 50, "x2": 160, "y2": 90}

# Hard-coded robot eef position constraints (mm)
robot_x_bounds = [485,180]
robot_y_bounds = [345,-210]
robot_z_bounds = [400,170]
boundary = robot_x_bounds + robot_y_bounds + robot_z_bounds

# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("R.txt")
t = np.loadtxt("t.txt")

# ---------------- GLOBAL STATE ----------------
arm = XArmAPI(XARM_IP)

mouse_pos = (0, 0)

uv = None
xyz = None
pose = None

last_click_time = 0

# ---------------- XARM ----------------
def setup_xarm():
        """Initialize XArm for control with safety limits."""
        if arm.get_state() != 0:
            arm.clean_error()
            time.sleep(0.5)

        arm.motion_enable(enable=True)
        arm.set_mode(0)
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
        arm.set_reduced_max_tcp_speed(200)  # mm/s
        arm.set_reduced_max_joint_speed(60) # deg/s
        arm.set_reduced_mode(True)

        print('XArm initialized with safety limits')


def control_callback():
    """Send end-effector control pose."""

    # Software boundary constraint to avoid triggering a controller error
    for i in range(len(pose)):
        pose_max = boundary[2*i]
        pose_min = boundary[2*i+1]
        if pose[i] > pose_max:
            pose[i] = pose_max - 1
        if pose[i] < pose_min:
            pose[i] = pose_min + 1


    arm.set_position(x=pose[0], y=pose[1], z=pose[2],
                    roll=-175, pitch=-5, yaw=-5,
                    is_radian=False, wait=True
                    )

# ---------------- MOUSE ----------------
def mouse_callback(event, x, y, flags, depth_frame):
    global mouse_pos, xyz, uv, pose, last_click_time

    mouse_pos = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # debounce clicks
    if time.time() - last_click_time < 0.25:
        return
    last_click_time = time.time()

    # --- BUTTON CLICK ---
    if SEND_BUTTON["x1"] <= x <= SEND_BUTTON["x2"] and SEND_BUTTON["y1"] <= y <= SEND_BUTTON["y2"]:
        print("send button clicked")
        if xyz is None:
            print("No active point.")
            return

        control_callback()
        return


    # --- VALIDATE PIXEL ---
    h, w = depth_frame.get_height(), depth_frame.get_width()
    if not (0 <= x < w and 0 <= y < h):
        return

    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        print("Invalid depth")
        return

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
    uv = [x,y]
    pose = R @ np.array(xyz)*1000 + t
    print(pose)


# ---------------- VIS ----------------
def create_colorbar(height, max_depth=2.0):
    gradient = np.linspace(max_depth, 0, height)
    gradient = (gradient / max_depth * 255).astype(np.uint8)
    gradient = np.tile(gradient[:, np.newaxis], (1, 50))

    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    for i, d in enumerate(np.linspace(0, max_depth, 6)):
        y = int(height * (i / 5))
        cv2.putText(colorbar, f"{max_depth - d:.1f}m",
                    (2, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)

    return colorbar

import numpy as np

# ---------------- MAIN ----------------
def main():
    setup_xarm()

    # start camera
    cfg = rs.config()
    cfg.enable_device(CAMERA_SERIAL)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipe = rs.pipeline()
    try:
        profile = pipe.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

    except RuntimeError as e:
        if not pipe:
            print("No cameras.")
            return

    align = rs.align(rs.stream.color)

    print("Press 'q' to quit.")

    try:
        while True:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)

            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            h, w, _ = color_image.shape

            # --- HOVER XYZ ---
            u, v = mouse_pos
            if 0 <= u < w and 0 <= v < h:
                depth = depth_frame.get_distance(u, v)
                if depth > 0:
                    intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    point = rs.rs2_deproject_pixel_to_point(intrin, [u, v], depth)
                    hover_text = f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}"
                else:
                    hover_text = "Invalid depth"
            else:
                hover_text = ""

            # --- DEPTH VIS ---
            depth_m = depth_image * depth_scale
            depth_clipped = np.clip(depth_m, 0, MAX_DEPTH_METERS)
            depth_norm = (depth_clipped / MAX_DEPTH_METERS * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            colorbar = create_colorbar(h, MAX_DEPTH_METERS)
            depth_display = np.hstack((depth_colormap, colorbar))

            combined = np.hstack((color_image, depth_display))

            # --- PANEL ---
            panel = np.zeros((combined.shape[0], 320, 3), dtype=np.uint8)

            y_offset = 30
            color = (0,255,255)

            if xyz is not None:
                txt1 = f"XYZ {xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}"
                cv2.putText(panel, txt1, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 20

                if pose is not None:
                    txt2 = f"   EEF {pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}"
                else:
                    txt2 = "   EEF: [not set]"

                cv2.putText(panel, txt2, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
                y_offset += 25

                combined = np.hstack((combined, panel))

            # --- DRAW CLICK POINTS ---
            if uv is not None:
                px, py = uv
                cv2.circle(combined, (px, py), 4, (0,255,0), -1)

            # --- SEND BUTTON ---
            cv2.rectangle(combined,
                            (SEND_BUTTON["x1"], SEND_BUTTON["y1"]),
                            (SEND_BUTTON["x2"], SEND_BUTTON["y2"]),
                            (0,255,0), -1)

            cv2.putText(combined, "Send Pose",
                        (SEND_BUTTON["x1"]+10, SEND_BUTTON["y1"]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,0), 2)

            # --- HOVER TEXT ---
            cv2.putText(combined, hover_text,
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,255), 2)
            
            name = "Test Calibration"

            cv2.imshow(name, combined)

            cv2.setMouseCallback(name, mouse_callback, depth_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()