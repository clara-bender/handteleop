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

POSE_BUTTON = {"x1": 10, "y1": HEIGHT-100, "x2": 160, "y2": HEIGHT-70}
SAVE_BUTTON = {"x1": 10, "y1": HEIGHT-50, "x2": 160, "y2": HEIGHT-20}

# ---------------- GLOBAL STATE ----------------
arm = XArmAPI(XARM_IP)

mouse_pos = (0, 0)

data_points = []       # [{ "xyz": [...], "pose": [...] or None }]
current_index = None   # active point index

last_click_time = 0


# ---------------- XARM ----------------
def setup_xarm():
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    time.sleep(1)


def state_callback():
    """Get current end-effector pose."""
    try:
        code, eef_pos = arm.get_position()
        if code != 0:
            print(f'Failed to get XArm position: {code}')
            return None

        # convert to radians for rotation
        angles_rad = (np.array(eef_pos[3:6]) * np.pi / 180).tolist()
        eef_pos = np.array(eef_pos[:3] + angles_rad, dtype=np.float32)

        return eef_pos

    except Exception as e:
        print(f'Error getting state: {e}')
        return None


# ---------------- MOUSE ----------------
def mouse_callback(event, x, y, flags, depth_frame):
    global mouse_pos, current_index, last_click_time

    mouse_pos = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # debounce clicks
    if time.time() - last_click_time < 0.25:
        return
    last_click_time = time.time()

    # --- BUTTON CLICK ---
    if POSE_BUTTON["x1"] <= x <= POSE_BUTTON["x2"] and POSE_BUTTON["y1"] <= y <= POSE_BUTTON["y2"]:
        if current_index is None:
            print("No active point.")
            return

        pose = state_callback()
        if pose is not None:
            data_points[current_index]["pose"] = pose
            print(f"[UPDATED] Point {current_index} pose: {pose}")
        return
    
    if SAVE_BUTTON["x1"] <= x <= SAVE_BUTTON["x2"] and SAVE_BUTTON["y1"] <= y <= SAVE_BUTTON["y2"]:
        if len(data_points) == 0:
            return

        P_list = []
        Q_list = []

        for i, item in enumerate(data_points):
            if item["pose"] is None:
                continue

            p = item["xyz"]
            q = item["pose"][:3]  # only translation part of EEF

            P_list.append(p)
            Q_list.append(q)

        if len(P_list) < 3:
            print("Need at least 3 valid pairs")
            return

        P = np.array(P_list, dtype=np.float32)
        Q = np.array(Q_list, dtype=np.float32)

        R,t = rigid_transform_3d(P, Q)

        # R: (n,n), t: (n,)
        np.savetxt("R.txt", R)
        np.savetxt("t.txt", t)

        print(R)
        print(t)

        data = np.hstack([P, Q])  # shape: (N, 6)

        np.savetxt(
            "calibration_pairs.csv",
            data,
            delimiter=",",
            header="px,py,pz,qx,qy,qz",
            comments=""
)

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
    point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)

    # --- CREATE NEW POINT ---
    data_points.append({
        "xyz": point,
        "pose": None,
        "pixel": [x,y]
    })

    current_index = len(data_points) - 1
    print(f"[NEW] Point {current_index}: {point}")


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

# ---------------- VIS ----------------
def rigid_transform_3d(P, Q):
    """
    P: Nx3 source points
    Q: Nx3 target points
    Returns: R, t such that Q ≈ R P + t
    """

    assert P.shape == Q.shape

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P

    return R, t


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
            for i, item in enumerate(data_points[-10:]):
                xyz = item["xyz"]
                pose = item["pose"]

                color = (0,255,255) if i == current_index else (0,255,0)

                txt1 = f"{i}: XYZ {xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}"
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
            for item in data_points:
                px, py = item["pixel"]
                cv2.circle(combined, (px, py), 4, (0,255,0), -1)

            # --- POSE BUTTON ---
            cv2.rectangle(combined,
                            (POSE_BUTTON["x1"], POSE_BUTTON["y1"]),
                            (POSE_BUTTON["x2"], POSE_BUTTON["y2"]),
                            (0,255,0), -1)

            cv2.putText(combined, "Get Pose",
                        (POSE_BUTTON["x1"]+10, POSE_BUTTON["y1"]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,0), 2)
            
            # --- SAVE BUTTON ---
            cv2.rectangle(combined,
                            (SAVE_BUTTON["x1"], SAVE_BUTTON["y1"]),
                            (SAVE_BUTTON["x2"], SAVE_BUTTON["y2"]),
                            (0,255,0), -1)

            cv2.putText(combined, "Save",
                        (SAVE_BUTTON["x1"]+10, SAVE_BUTTON["y1"]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,0), 2)


            # --- HOVER TEXT ---
            cv2.putText(combined, hover_text,
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,255), 2)

            cv2.imshow("Hand camera", combined)

            cv2.setMouseCallback("Hand camera", mouse_callback, depth_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()