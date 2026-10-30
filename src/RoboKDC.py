'''
This file serves to store the most common functions for Robotic Kinematics Dynamics and Control
'''

import numpy as np

def RotZ(theta: float) -> np.ndarray:
    R = [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]]
    R = np.array(R)
    return R

def RotY(theta: float) -> np.ndarray:
    R = [[np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]]
    R = np.array(R)
    return R

def RotX(theta: float) -> np.ndarray:
    R = [[0, 0, 1],
         [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]]
    R = np.array(R)
    return R