# -*- coding: utf-8 -*-
"""
Calculates the possible planes that could make up the etched triangular shapes

Created on Thu May 12 11:29:39 2016

@author: Martin Friedl
"""

import math

import numpy as np


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


def find_angles_between_two_vectors(v1, v2):
    """
    Returns the angle between two vectors
    :param v1: vector 1 as a list
    :param v2: vector 2 as a list
    :return:
    """
    v1dv2 = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not (np.isfinite(v1dv2 / norm)):
        return 0
    angle = np.rad2deg(np.arccos(v1dv2 / norm))
    if np.isnan(angle):
        return 0
    elif angle > 90:
        return 180.0 - angle
    else:
        return angle


def find_possible_planes2(candidates, ref_plane, angle_range):
    """
    Finds the possible planes that have an angle within the range with the reference plane
    :param candidates: list of planes to test
    :param ref_plane: reference plane to test against
    :param angle_range: range of angles in which to keep the candidates
    :return: filtered list of candidate planes
    """
    min_angle = angle_range[0]
    max_angle = angle_range[1]
    if min_angle > max_angle:
        ValueError('Range must be of format [min_angle, max_angle]')

    new_candidates = []
    for candidate in candidates:
        if min_angle <= find_angles_between_two_vectors(candidate, ref_plane) <= max_angle:
            new_candidates.append(candidate)

    return new_candidates


# Define miller index search range
srange = 3

# Create a big array of possible planes to search
all_planes = []
for x in range(-srange, srange + 1):
    for y in range(-srange, srange + 1):
        for z in range(-srange, srange + 1):
            all_planes.append([x, y, z])

# Look for all planes that make the angles measured in AFM with the (-1-1-1) plane
candidates1 = find_possible_planes2(all_planes, [-1, -1, -1], [34, 43])
# Look for all planes that are 'parallel' to the(11-2) direction
vect_90deg_to_112 = list(np.dot(rotation_matrix([-1, -1, -1], -np.pi / 2.), [-1, -1, 2]))
candidates2 = find_possible_planes2(candidates1, vect_90deg_to_112, [89, 91])

print('Possible planes are:')
for plane in candidates2:
    print('\t{} angle with (111) = {:.2f}'.format(str(plane), find_angles_between_two_vectors([1, 1, 1], plane)))
