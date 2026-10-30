import RoboKDC as ro
import numpy as np

# Check RPY
rpyRot = np.array([1.8901,2.4888,0.6128])
rpyRot_check = np.array(
    [[-0.6499, 0.6523,0.3901],
     [-0.4569,0.0749,-0.8864],
     [-0.6074,-0.7543,0.2494]]
)
rpyRot_ans = ro.rpy2Rot(1.8901,2.4888,0.6128)
print("Check rpy2Rot:")
print(np.max(np.abs(rpyRot_check-rpyRot_ans)))

RotRPY = np.array(
    [[0.4421,0.5884,-0.6770],
     [0.4000,-0.8049,-0.4383],
     [-0.8028,-0.0770,-0.5913]]
)
angles_check = np.array(
    [[-3.0121,0.1295],
     [0.9320,2.2096],
     [0.7355,-2.4061]]
)
angles_ans = ro.rot2RPY(RotRPY)
print("Check rot2RPY:")
print(np.max(np.abs(angles_check-angles_ans)))
print("")

# Check angle axis
R_aa = np.array(
    [[0.0774,0.7739,0.6286],
     [-0.2038, -0.6049, 0.7698],
     [0.9760, -0.1877, 0.1109]]
)
AA_check = np.array([[-1.5991],[-0.5801],[-1.6328]])
AA_ans = ro.rot2AngleAxis(R_aa)
print("Check rot2AngleAxis:")
print(np.max(np.abs(AA_check - AA_ans)))
aa2R = np.array([[0.0920],[0.1469],[-0.1420]])
RAA_check = np.array(
    [[0.979212227141297,0.147572479401704,0.139160977074089],
     [-0.134114747359959,0.985741654123937,-0.101620007211070],
     [-0.152173088151473,0.080844014300610,0.985041926312889]]
)
RAA_ans = ro.angleAxis2Rot(aa2R)
print("Check angleAxis2Rot:")
print(np.max(np.abs(RAA_check-RAA_ans)))
print("")

# Check quaternions
R_q = np.array(
    [[-0.7425,-0.4211,-0.5210],
     [-0.6278,0.7087,0.3219],
     [0.2337,0.5661,-0.7905]]
)
Q_check = np.array([[0.2096],[0.2913],[-0.9002],[-0.2466]])
Q_ans = ro.rot2Quat(R_q)
print("Check rot2Quat:")
print(np.max(np.abs(Q_check-Q_ans)))
Q_r = np.array([[0.7256],[-0.2644],[0.5185],[0.3670]])
RQ_check = np.array(
    [[0.1929, -0.8068, 0.5585],
     [0.2584, 0.5909, 0.7643],
     [-0.9466,-0.0031,0.3225]]
)
RQ_ans = ro.quat2Rot(Q_r)
print("Check quat2Rot:")
print(np.max(np.abs(RQ_check-RQ_ans)))
q_in = np.array([[0.3710],[0.7702],[0.3730],[0.3606]])
p_in = np.array([[0.7602],[0.2958],[0.5486],[0.1831]])
pq_check = np.array([[-0.2165],[0.5657],[0.4528],[0.6543]])
pq_ans = ro.multiplyQuat(q_in,p_in)
print("Check multiplyQuat:")
print(np.max(np.abs(pq_check-pq_ans)))
print("")

# Test dual quaternion functions
pos = np.array([[-0.2484],[0.4124],[0.9332]])
quat = np.array([[0.5905],[0.4038],[0.6107],[-0.3395]])
dpdq_check = np.array(
    [[0.5905],[0.4038],[0.6107],[-0.3395],
     [0.0827],[-0.4283],[0.2680],[0.1165]]
)
dpdq_ans = ro.pq2DualQuat(pos,quat)
print("Check pq2DualQuat:")
print(np.max(np.abs(dpdq_check-dpdq_ans)))
dqdp = np.array(
    [[-0.8036],[0.5495],[-0.0876],[-0.2114],
     [-0.7960],[-0.9497],[-0.5308],[0.7773]]
)
pq_check = np.array(
    [[2.0405],[0.2609],[-2.3353],
     [-0.8036],[0.5495],[-0.0876],[-0.2114]]
)
pq_ans = ro.dualQuat2PQ(dqdp)
print("Check dualQuat2PQ:")
print(np.max(np.abs(pq_check-pq_ans)))
dq_left = np.array(
    [[-0.6378],[-0.4730],[-0.6054],[-0.0543],
     [0.0857],[-0.3596],[0.1652],[0.2838]]
)
dq_right = np.array(
    [[-0.6686],[-0.6297],[-0.3909],[-0.0604],
     [-0.3453],[0.8970],[-0.9925],[0.8943]]
)
dq_check = np.array(
    [[-0.1114],[0.7332],[0.6597],[-0.1215],
     [-0.1098],[-0.7166],[0.8720],[0.5106]]
)
dq_ans = ro.multiplyDualQuat(dq_left,dq_right)
print("Check multiplyDualQuat:")
print(np.max(np.abs(dq_check-dq_ans)))
print("")

# Check dhTransform
dh_check = np.array(
    [[-0.8841, -0.1785, -0.4318, -1.4545],
     [0.4673,-0.3378,-0.8171,0.7688],
     [0, -0.9242, 0.3820,-1.3827],
     [0,0,0,1]]
)
dh_ans = ro.dhTransform(1.6452,-1.3827,-1.1788,2.6554)
print("Check dhTransform:")
print(np.max(np.abs(dh_check-dh_ans)))