import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import axangle2quat, qmult, rotate_vector, qinverse, quat2mat, nearly_equivalent
import math
from sklearn.preprocessing import normalize

######################################################
# Inputs to the program
ZA = np.array([-1, 0, -1])  # Current zone-axis
D = np.array([0., 1, 0])  # Known direction in this zone axis (from diffraction pattern)
DA = -10  # Angle to the vertical of the known direction (degrees, clockwise)
ZA2 = np.array([1, 1, -1])  # Desired zone axis
#####################################################

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R_mat):
    Rt = np.transpose(R_mat)
    shouldBeIdentity = np.dot(Rt, R_mat)
    I = np.identity(3, dtype=R_mat.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def angle_between_vectors(v1, v2):
    """
    Calculates the (smallest) angle between two vectors
    :param v1: First vector
    :param v2: Second vector
    :return: angle in radians
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_quaternion(lst1, lst2, matchlist=None):
    """
    Source: https://stackoverflow.com/questions/16648452/calculating-quaternion-for-transformation-between-2-3d-cartesian-coordinate-syst
    :param lst1: list of numpy.arrays where every array represents a 3D vector.
    :param lst2: list of numpy.arrays where every array represents a 3D vector.
    :param matchlist: The optional matchlist argument is used to tell the function which point of lst2 should be transformed to which point in lst1
    :return: quaternion for the coordinate transformation
    """
    if not matchlist:
        matchlist = range(len(lst1))
    M = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i, coord1 in enumerate(lst1):
        x = np.matrix(np.outer(coord1, lst2[matchlist[i]]))
        M = M + x

    N11 = float(M[0][:, 0] + M[1][:, 1] + M[2][:, 2])
    N22 = float(M[0][:, 0] - M[1][:, 1] - M[2][:, 2])
    N33 = float(-M[0][:, 0] + M[1][:, 1] - M[2][:, 2])
    N44 = float(-M[0][:, 0] - M[1][:, 1] + M[2][:, 2])
    N12 = float(M[1][:, 2] - M[2][:, 1])
    N13 = float(M[2][:, 0] - M[0][:, 2])
    N14 = float(M[0][:, 1] - M[1][:, 0])
    N21 = float(N12)
    N23 = float(M[0][:, 1] + M[1][:, 0])
    N24 = float(M[2][:, 0] + M[0][:, 2])
    N31 = float(N13)
    N32 = float(N23)
    N34 = float(M[1][:, 2] + M[2][:, 1])
    N41 = float(N14)
    N42 = float(N24)
    N43 = float(N34)

    N = np.matrix([[N11, N12, N13, N14],
                   [N21, N22, N23, N24],
                   [N31, N32, N33, N34],
                   [N41, N42, N43, N44]])

    values, vectors = np.linalg.eig(N)
    w = list(values)
    mw = max(w)
    quat = vectors[:, w.index(mw)]
    quat = np.array(quat).reshape(-1, ).tolist()
    return quat


x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

DA = DA * np.pi / 180.  # Convert from rad to degrees
ZA = ZA / np.linalg.norm(ZA)  # Normalize
D = D / np.linalg.norm(D)  # Normalize
ZA2 = ZA2 / np.linalg.norm(ZA2)  # Normalize

# Calculate the rotation matrix to transform from TEM coordinates to crystal coordinates
# Define 3 directions in TEM coordinate system
z1 = z_axis
y1 = np.array([np.sin(DA), np.cos(DA), 0])
x1 = np.cross(z1, y1)
z1 = z1 / np.linalg.norm(z1)
y1 = y1 / np.linalg.norm(y1)
x1 = x1 / np.linalg.norm(x1)

# Define the 3 directions in the crystal coordinate system
z2 = ZA
y2 = D
x2 = np.cross(z2, y2)
z2 = z2 / np.linalg.norm(z2)
y2 = y2 / np.linalg.norm(y2)
x2 = x2 / np.linalg.norm(x2)

# Transformation from crystal zone axis to new zone axis
angle = angle_between_vectors(ZA, ZA2)
axis = np.cross(ZA, ZA2)
R2 = rotation_matrix(axis, angle)
R2 = normalize(R2)

z2r = ZA2
y2r = np.dot(R2, y2)
x2r = np.dot(R2, x2)
z2r = z2r / np.linalg.norm(z2r)
y2r = y2r / np.linalg.norm(y2r)
x2r = x2r / np.linalg.norm(x2r)

q_12 = get_quaternion([x1, y1, z1], [x2, y2, z2])
q_21 = qinverse(q_12)

B_1to2 = normalize(quat2mat(q_12))
B_2to1 = normalize(quat2mat(q_21))

R1 = np.dot(B_2to1, np.dot(R2, B_1to2))
R1 = normalize(R1)

# print np.dot(R2, np.vstack((X1, Y1, Z1)))

rot_angles = -rotationMatrixToEulerAngles(R1) * 180 / np.pi
rot_angles = np.round(rot_angles, 2)
print("Tilt around x by {:.2f} degrees".format(rot_angles[0]))
print("Then tilt around y by {:.2f} degrees".format(rot_angles[1]))
