from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import time
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
SRC_REPO = "clara/glove_placing_hand_waving"
DST_REPO = "clara/glove_placing_hand_waving_everything"

SRC_DATASET = HF_LEROBOT_HOME/SRC_REPO
DST_DATASET = HF_LEROBOT_HOME/DST_REPO

TASK_DESCRIPTION = "Follow the hand or put the glove in the lower left corner of the workspace."
# TASK_DESCRIPTION = "Move the glove to the lower lefthand corner of the workspace."
og_dataset = LeRobotDataset(
        root=SRC_DATASET,
        repo_id=SRC_REPO,
    )

# ============================================================
# Remove existing destination
# ============================================================

if DST_DATASET.exists():
    shutil.rmtree(DST_DATASET)

# ============================================================
# Create folders
# ============================================================

new_dataset = LeRobotDataset.create(
        repo_id=DST_REPO,
        robot_type="xarm",
        fps=og_dataset.fps,
        features=og_dataset.features,
    )


# Load default hand and robot frame images
hand_img_blank_BGR = cv2.imread("hand_frame_blank_color.png")
hand_img_blank = cv2.cvtColor(hand_img_blank_BGR, cv2.COLOR_BGR2RGB)

robot_img_blank_BGR = cv2.imread("robot_frame_blank.png")
robot_img_blank = cv2.cvtColor(robot_img_blank_BGR, cv2.COLOR_BGR2RGB)

def decode_img(x):
    if isinstance(x, dict):
        x = x.get("bytes", None)

    if x is None:
        return None

    arr = np.frombuffer(x, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR).copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def fix_vec(x, dim=None):
    if isinstance(x, dict):
        x = x.get("array", x.get("value", x))

    x = np.array(x, dtype=np.float32)

    if dim is not None:
        x = x.reshape(dim)

    return x

def zero_vec(x, dim=None):
    if isinstance(x, dict):
        x = x.get("array", x.get("value", x))

    x = np.array(x, dtype=np.float32)
    x = np.zeros_like(x)

    if dim is not None:
        x = x.reshape(dim)

    return x

# ============================================================
# Process parquet files
# ============================================================

count = 0

for parquet_path in sorted((SRC_DATASET /"data"/ "chunk-000").glob("*.parquet")):

    # if count > 45:
    #     break

    print(f"Processing {parquet_path.name}")

    df = pd.read_parquet(parquet_path)
    # -------------------------------------------------------
    # Alternate blackout
    # --------------------------------------------------------

    for i in range(len(df)):

        robot_img = decode_img(df.at[i, "robot_camera"])
        hand_img = decode_img(df.at[i, "hand_camera"])
        extra_img = decode_img(df.at[i, "extra_camera"])

        # if count <= 127:
        #     random_number = np.random.rand(1)
        #     if random_number > 0.2:
        #         hand_img = hand_img_blank

        # robot_img = np.zeros_like(robot_img)

        # if i % 3 == 0:
        #     continue
        # elif (i % 3) == 1:
        #     robot_img = robot_img_blank
        # else:
        #     hand_img = hand_img_blank


    # --------------------------------------------------------
    # Save modified parquet
    # --------------------------------------------------------

        

        observation = {
            "eef_position": fix_vec(df.at[i, "eef_position"], 6),
            "gripper_position": fix_vec(df.at[i, "gripper_position"], 1),
            "actions": fix_vec(df.at[i, "actions"], 7),

            "robot_camera": robot_img,
            "hand_camera": hand_img,
            "extra_camera": extra_img,

            "task": TASK_DESCRIPTION,
            }
        
        new_dataset.add_frame(observation)

    count +=1

    new_dataset.save_episode()

print("Done.")
