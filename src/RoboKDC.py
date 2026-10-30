'''
This file serves to store the most common functions for Robotic Kinematics Dynamics and Control
'''

# TODO: add function help to each function

import numpy as np

def rotZ(theta: float) -> np.ndarray:
    R = [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]]
    R = np.array(R)
    return R

def rotY(theta: float) -> np.ndarray:
    R = [[np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]]
    R = np.array(R)
    return R

def rotX(theta: float) -> np.ndarray:
    R = [[1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]]
    R = np.array(R)
    return R

def rpy2Rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    R = rotZ(yaw) @ rotY(pitch) @ rotX(roll)
    return R

def rot2RPY(R: np.ndarray) -> np.ndarray:
    tol = 1e-6
    th1 = np.atan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    th2 = np.atan2(-R[2,0], -np.sqrt(R[0,0]**2 + R[1,0]**2))
    th = np.zeros([1,2])
    th[0,0] = th1; th[0,1] = th2
    psi = np.zeros([1,2])
    phi = np.zeros([1,2])
    for i in range(0,2):
        c_th = np.cos(th[0,i])
        if abs(c_th) < tol:
            psi[0,i] = 0 # Assume yaw = 0 in degeneracy case
            if th[0,i] > 0:
                # Pitch = +pi/2
                # R(1,2) -> sin(phi), R(1,3) -> cos(phi)
                phi[0,i] = np.atan2(R[0,1],R[0,2])
            else:
                # Pitch = -pi/2
                # Flipped angles
                phi[0,i] = np.atan2(-R[0,1],-R[0,2])
        else:
            # General Case
            phi[0,i] = np.atan2(R[2,1]/c_th, R[2,2]/c_th)
            psi[0,i] = np.atan2(R[1,0]/c_th, R[0,0]/c_th)
    rpy = np.zeros([3,2])
    rpy[[0],:] = phi; rpy[[1],:] = th; rpy[[2],:] = psi
    return rpy

def cpMap(w: np.ndarray) -> np.ndarray:
    w = np.ravel(w)
    X = [[0, -w[2], w[1]],
         [w[2], 0, -w[0]],
         [-w[1], w[0], 0]]
    return np.array(X)

def rot2AngleAxis(R: np.ndarray) -> np.ndarray:
    tol = 1e-6
    # Find theta
    c_th = (np.trace(R)-1)/2.0
    c_th = max(min(c_th, 1),-1) # clamp c_th for numerical stability
    thSol = np.acos(c_th)
    khat = np.zeros([3,1])

    # Handle small angle singularities
    if thSol < tol:
        denom = 2.0
        # Solve for khat
        khat = 1/(denom) * np.array([[R[2,1]-R[1,2]],[R[0,2]-R[2,0]],[R[1,0]-R[0,1]]])
        Omega = khat
        return np.array(Omega)
    elif abs(thSol-np.pi) < tol or abs(np.trace(R)+1) < tol:
        # Handle pi multiples
        # Get largest diagonal
        i = np.argmax([[R[2,1]-R[1,2]],[R[0,2]-R[2,0]],[R[1,0]-R[0,1]]])
        if i == 1:
            khat[0] = np.sqrt((R[0,0] + 1)/2)
            khat[1] = R[0,1] / (2*khat[0])
            khat[2] = R[0,2] / (2*khat[0])
        elif i == 2:
            khat[1] = np.sqrt((R[1,1] + 1)/2)
            khat[0] = R[0,1] / (2*khat[1])
            khat[2] = R[1,2] / (2*khat[1])
        else:
            khat[2] = np.sqrt((R[2,2] + 1)/2)
            khat[0] = R[0,2] / (2*khat[2])
            khat[1] = R[1,2] / (2*khat[2])
        Omega = thSol*khat
        return np.array(Omega)
    else: # General case
        denom = 2*np.sin(thSol)
        khat = 1/(denom) * np.array([[R[2,1]-R[1,2]],[R[0,2]-R[2,0]],[R[1,0]-R[0,1]]])
        Omega = thSol*khat
        return np.array(Omega)

def angleAxis2Rot(Omega: np.ndarray) -> np.ndarray:
    tol = 1e-6
    Omega = np.ravel(Omega)
    # Angle is the magnitude of Omega (Omega = th*khat)
    th = np.linalg.norm(Omega)
    # Deal with zero vector edge case -> identity matrix
    if th < tol:
        R = np.eye(3)
        return R
    # Otherwise do general case
    khat = Omega/th
    khat = khat/np.linalg.norm(khat) # force normality
    k1 = khat[0]; k2 = khat[1]; k3 = khat[2]
    R = [[(k1**2)*(1-np.cos(th))+np.cos(th), k1*k2*(1-np.cos(th))-k3*np.sin(th), k1*k3*(1-np.cos(th))+k2*np.sin(th)],
        [k1*k2*(1-np.cos(th))+k3*np.sin(th), (k2**2)*(1-np.cos(th))+np.cos(th), k2*k3*(1-np.cos(th))-k1*np.sin(th)],
        [k1*k3*(1-np.cos(th))-k2*np.sin(th), k2*k3*(1-np.cos(th))+k1*np.sin(th), (k3**2)*(1-np.cos(th))+np.cos(th)]]
    return np.array(R)

def rot2Quat(R: np.ndarray) -> np.ndarray:
    tR = np.trace(R)
    if tR > 0:
        q0 = np.sqrt((1+tR)/4)
        q1 = (R[2,1]-R[1,2])/(4*q0)
        q2 = (R[0,2]-R[2,0])/(4*q0)
        q3 = (R[1,0]-R[0,1])/(4*q0)
    else:
        # Get largest diagonal elements and avoid divide by zero erros
        i = np.argmax([R[0,0],R[1,1],R[2,2]])
        if i == 1:
            q1 = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])/2.0
            q0 = (R[2,1] - R[1,2])/(4*q1)
            q2 = (R[0,1] + R[1,0])/(4*q1)
            q3 = (R[0,2] + R[2,0])/(4*q1)
        elif i == 2:
            q2 = np.sqrt(1 - R[0,0] + R[1,1] - R[2,2])/2.0
            q0 = (R[0,2] - R[2,0])/(4*q2)
            q1 = (R[0,1] + R[1,0])/(4*q2)
            q3 = (R[1,2] + R[2,1])/(4*q2)
        else:
            q3 = np.sqrt(1 - R[0,0] - R[1,1] + R[2,2])/2.0
            q0 = (R[1,0] - R[0,1])/(4*q3)
            q1 = (R[0,2] + R[2,0])/(4*q3)
            q2 = (R[1,2] + R[2,1])/(4*q3)
    Q = np.array([[q0],[q1],[q2],[q3]])
    Q = Q/np.linalg.norm(Q) # Force normality to account for error accum
    return np.array(Q)

def quat2Rot(Q: np.ndarray) -> np.ndarray:
    Q = np.ravel(Q)
    q0 = Q[0]; q1 = Q[1]; q2 = Q[2]; q3 = Q[3]
    R = [[1-2*(q2**2)-2*(q3**2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
        [2*q1*q2+2*q0*q3, 1-2*(q1**2)-2*(q3**2), 2*q2*q3-2*q0*q1],
        [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*(q1**2)-2*(q2**2)]]
    return np.array(R)

def transform2Twist(H: np.ndarray) -> np.ndarray:
    tol = 1e-6
    R = H[1:3,1:3]; d = H[1:3,4]
    Omega = rot2AngleAxis(R)
    th = np.linalg.norm(Omega)
    if th < tol: # Pure translation case
        v = d
    else: # General case
        khat = Omega/th
        khat = khat/np.linalg.norm(khat) # Force normality
        v = (
            np.sin(th)/(2*(1-np.cos(th))) * np.eye(3) + 
            (2*(1-np.cos(th))-th*np.sin(th))/(2*th*(1-np.cos(th))) * 
            (khat @ np.transpose(khat)) - 1/2*cpMap(khat)
        ) @ d
    t = np.zeros([6,1])
    t[0:2] = v; t[3:5] = Omega
    return np.array(t)

def twist2Transform(t: np.ndarray) -> np.ndarray:
    t = np.ravel(t)
    tol = 1e-6
    v = t[0:2]; Omega = t[3:5]
    H = np.eye(4)
    th = np.linalg.norm(Omega)
    khat = Omega/th
    khat = khat/np.linalg.norm(khat) # Force normality
    # Pure translation case
    if th < tol:
        R = np.eye(3)
        d = v
    else: # General case
        R = (
            np.eye(3)*np.cos(th)+np.sin(th)*cpMap(khat)+
            (1-np.cos(th))*(khat @ np.transpose(khat))
        )
        d = ((np.eye(3)-R) @ cpMap(khat) + Omega @ np.transpose(khat)) @ v
    H[0:2,0:2] = R
    H[1:3,4] = d
    return H

def multiplyQuat(Q_left: np.ndarray, Q_right: np.ndarray) -> np.ndarray:
    q = np.zeros([3,1]); p = np.zeros([3,1])
    q0 = Q_left[0]; q[:,[0]] = Q_left[1:]
    p0 = Q_right[0]; p[:,[0]] = Q_right[1:]
    q_out0 = p0*q0 - q.T @ p
    q_im = p0*q + q0*p - cpMap(p) @ q
    q_out = np.zeros([4,1])
    q_out[0] = q_out0
    q_out[1:,[0]] = q_im
    return q_out

def pq2DualQuat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    tol = 1e-6
    Q = np.zeros([8,1]) # Flatten array
    t = pos.reshape(-1,1); H = quat.reshape(-1,1)
    T = np.zeros([4,1]); T[1:4] = t[:]
    sin_half_th = np.linalg.norm(H[1:4])
    if np.linalg.norm(t) < tol: # No translation case
        Q[0:4] = H
    elif sin_half_th < tol: # No rotation case
        Q[0:4] = np.array([[1],[0],[0],[0]]) # Unit Quaternion
        Q[5:] = t/2.0 # Zero real part
    else: # General case
        Q[0:4] = H
        Q[4:] = multiplyQuat(T,H)/2.0
    return Q

def dualQuat2PQ(dual_quat: np.ndarray) -> np.ndarray:
    dual_quat = dual_quat.reshape(-1,1)
    P = dual_quat[0:4].copy()
    Q = dual_quat[4:]
    quat = P.copy()
    pConj = P.copy(); pConj[1:] = -pConj[1:]
    T = 2*multiplyQuat(Q,pConj)
    pos = T[1:4]
    pq = np.zeros([7,1])
    pq[0:3] = pos; pq[3:] = quat
    return pq

def multiplyDualQuat(dq_left: np.ndarray, dq_right: np.ndarray) -> np.ndarray:
    dq_left = dq_left.reshape(-1,1); dq_right = dq_right.reshape(-1,1)
    P_left = dq_left[0:4]; P_right = dq_right[0:4] # rotation quats
    Q_left = dq_left[4:]; Q_right = dq_right[4:]
    rot = multiplyQuat(P_left,P_right)
    disp = multiplyQuat(P_left,Q_right) + multiplyQuat(Q_left,P_right)
    dual_quat = np.zeros([8,1])
    dual_quat[0:4] = rot
    dual_quat[4:] = disp
    return dual_quat

def dhTransform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    # Initialize all 4 4x4 terms
    term1 = np.eye(4); term2 = np.eye(4)
    term3 = np.eye(4); term4 = np.eye(4)
    # Insert appropriate translation or rotation into blank 4x4 term
    term1[0:3,[3]] = np.array([[0],[0],[d]])
    term2[0:3,0:3] = rotZ(theta)
    term3[0:3,[3]] = np.array([[a],[0],[0]],ndmin=2)
    term4[0:3,0:3] = rotX(alpha)
    # Combine all terms
    H = term1 @ term2 @ term3 @ term4
    return H