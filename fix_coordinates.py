import numpy as np
import pandas as pd

df = pd.read_csv("calibration_pairs.csv")

P = df[["px","py","pz"]].to_numpy()  # camera
Q = df[["qx","qy","qz"]].to_numpy()  # robot

P = P * 1000.0

centroid_P = P.mean(axis=0)
centroid_Q = Q.mean(axis=0)

P_centered = P - centroid_P
Q_centered = Q - centroid_Q

H = P_centered.T @ Q_centered
U, S, Vt = np.linalg.svd(H)

R = Vt.T @ U.T

if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

t = centroid_Q - R @ centroid_P

P_pred = (R @ P.T).T + t
rmse = np.sqrt(np.mean(np.sum((P_pred - Q)**2, axis=1)))

print("RMSE (mm):", rmse)

np.savetxt("R.txt", R)
np.savetxt("t.txt", t)