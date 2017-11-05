import numpy as np
from transforms3d.quaternions import quat2mat

# TODO: Using polar coordinates would be much more elegant...would only need 2x2 rotation matrix

"""
Introduction
In the TEM it can be hard to find the proper zone axis that we are interested in. Often, we can get to some initial 
zone axis, but then to get to one that we are interested in, we have to navigate reciprocal space (by tilting) while
compensating for drifting of the sample out of the electron beam etc. This makes finding the zone axis that we are
interested in very long and tedious, sometimes seemingly impossible.

This tool uses math to try and make this process much quicker and convenient. The idea is that the user will have found 
some initial known zone axis and will have identified the relevant diffraction spots (directions) in the crystal. This 
unambiguously defines the orientation of the crystal in 3D space. Knowing this, the user inputs the zone axis that they 
would like to go to and this program computes the required alpha and beta tilt to reach the desired zone axis. The user
can then use these as "directions" to find the zone axis they are interested in.

The inputs to the code are the initial known zone axis, a known crystal direction, the (clockwise) angle from vertical
to this known direction (taken from a diffraction pattern) and finally the zone axis that they would like to reach.

The outputs are the angle in alpha and beta to tilt the specimen to reach the desired zone axis.
"""

######################################################
# Inputs to the program
ZA = np.array([-1, 0, -1])  # Current zone-axis
D = np.array([0., 1, 0])  # Known direction in this zone axis (from diffraction pattern)
DA = -10  # Angle to the vertical of the known direction (degrees, clockwise)
ZA2 = np.array([1, 1, -1])  # Desired zone axis


#####################################################

def is_rot_matrix(r_mat):
    """
    Checks if a matrix is a valid rotation matrix.
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    :param r_mat: Matrix
    :return: True if the matrix is a valid rotation matrix, False otherwise
    """
    r_t = np.transpose(r_mat)
    should_be_identity = np.dot(r_t, r_mat)
    i = np.identity(3, dtype=r_mat.dtype)
    n = np.linalg.norm(i - should_be_identity)
    return n < 1e-6


def matrix_to_euler(r_mat):
    """
    Calculates rotation matrix to euler angles
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    :param r_mat: Rotation matrix
    :return: List of static XYZ Euler angles (in radians) as numpy array
    """
    assert (is_rot_matrix(r_mat))

    sy = np.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(r_mat[2, 1], r_mat[2, 2])
        y = np.arctan2(-r_mat[2, 0], sy)
        z = np.arctan2(r_mat[1, 0], r_mat[0, 0])
    else:
        x = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
        y = np.arctan2(-r_mat[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix(rot_axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Source: https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    rot_axis = np.asarray(rot_axis)
    rot_axis = rot_axis / np.sqrt(np.dot(rot_axis, rot_axis))
    a = np.cos(theta / 2.0)
    b, c, d = -rot_axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def angle_between_vectors(v1, v2):
    """
    Calculates the (smallest) angle between two vectors
    Source: https://stackoverflow.com/questions/39497496/angle-between-two-vectors-3d-python
    :param v1: First vector
    :param v2: Second vector
    :return: angle in radians
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_quaternion(lst1, lst2, match_list=None):
    """
    Given two lists of 3 3d vectors, returns quaternion that transforms from one coordinate system to the other
    Source: https://stackoverflow.com/questions/16648452/calculating-quaternion-for-transformation-between-2-3d-cartesian-coordinate-syst
    :param lst1: list of numpy.arrays where every array represents a 3D vector.
    :param lst2: list of numpy.arrays where every array represents a 3D vector.
    :param match_list: (optional) Tells the function which point of lst2 should be transformed to which point in lst1
    :return: quaternion for the coordinate transformation
    """
    if not match_list:
        match_list = range(len(lst1))
    m = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i, coord1 in enumerate(lst1):
        x = np.matrix(np.outer(coord1, lst2[match_list[i]]))
        m = m + x

    n11 = float(m[0][:, 0] + m[1][:, 1] + m[2][:, 2])
    n22 = float(m[0][:, 0] - m[1][:, 1] - m[2][:, 2])
    n33 = float(-m[0][:, 0] + m[1][:, 1] - m[2][:, 2])
    n44 = float(-m[0][:, 0] - m[1][:, 1] + m[2][:, 2])
    n12 = float(m[1][:, 2] - m[2][:, 1])
    n13 = float(m[2][:, 0] - m[0][:, 2])
    n14 = float(m[0][:, 1] - m[1][:, 0])
    n21 = float(n12)
    n23 = float(m[0][:, 1] + m[1][:, 0])
    n24 = float(m[2][:, 0] + m[0][:, 2])
    n31 = float(n13)
    n32 = float(n23)
    n34 = float(m[1][:, 2] + m[2][:, 1])
    n41 = float(n14)
    n42 = float(n24)
    n43 = float(n34)

    n = np.matrix([[n11, n12, n13, n14],
                   [n21, n22, n23, n24],
                   [n31, n32, n33, n34],
                   [n41, n42, n43, n44]])

    values, vectors = np.linalg.eig(n)
    w = list(values)
    mw = max(w)
    quaternion = vectors[:, w.index(mw)]
    quaternion = np.array(quaternion).reshape(-1, ).tolist()
    return quaternion


# Declare some constants
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

# Convert angle from degrees to radians
DA = DA * np.pi / 180.

# Define the 3 directions in the TEM coordinate system
z1 = z_axis
y1 = np.array([np.sin(DA), np.cos(DA), 0])
x1 = np.cross(z1, y1)

# Define the 3 directions in the crystal coordinate system
z2 = ZA
y2 = D
x2 = np.cross(z2, y2)

# Determine rotation matrix to get from current zone axis to new zone axis (in the crystal's coordinate system)
angle = angle_between_vectors(ZA, ZA2)
axis = np.cross(ZA, ZA2)
R2 = rotation_matrix(axis, angle)

# Determine quaternion that transforms from TEM coordinate system to crystal coordinate system, convert it to rot matrix
B_1to2 = quat2mat(get_quaternion([x1, y1, z1], [x2, y2, z2]))
B_2to1 = np.linalg.inv(B_1to2)  # Calculate the inverse transformation

# Calculate rotation matrix in TEM coordinates
# by using change of basis matrices and previous rotation matrix in crystal coordinates
R1 = np.dot(B_2to1, np.dot(R2, B_1to2))

# Get the Euler angles from this new rotation matrix
rot_angles = -matrix_to_euler(R1) * 180 / np.pi
rot_angles = np.round(rot_angles, 2)

# Output required tilt information to the user
print("Tilt around x by {:.2f} degrees".format(rot_angles[0]))
print("Then tilt around y by {:.2f} degrees".format(rot_angles[1]))
