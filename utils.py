import math
from typing import List, Tuple
#
import cv2
import numpy as np


def determine_reference_axis_from_polygon(
        polygons: List[List[float]],
        line_length: int = 200
) -> Tuple[float, float, float, float, float, float]:
    """ Determine the 2D axis who x axis is parallel to the horizontal lines formed by `polygons`.

    Args:
    ---
    - `polygons`: List[List[float,float]]
        The 4 points in xy format representing the four corners of the polygon.
    - `line_length`: int
        The length of the axis.

    Returns:
    ---
        The points that can be used to draw two perpendicular lines forming the axis.
        (center_x, center_y, x2, y2, x3, y3)
    """
    rrect = cv2.minAreaRect(np.array(polygons, dtype=np.float32))
    (center_x, center_y), (width, height), angle = rrect

    if width < height:
        angle = angle + 90

    theta_rad = math.radians(angle)
    dx = math.cos(theta_rad)
    dy = math.sin(theta_rad)

    if width < height:
        x2 = int(center_x - (line_length / 2) * dx)
        y2 = int(center_y - (line_length / 2) * dy)
    else:
        x2 = int(center_x + (line_length / 2) * dx)
        y2 = int(center_y + (line_length / 2) * dy)

    # find a third pts which forms a line that is perpendicular to the
    # line formed by (cx,cy) and (x2,y2)
    v = np.array([x2-center_x, y2-center_y])
    v_norm = v/math.sqrt(v[0]**2+v[1]**2)
    v_perp = np.array([v_norm[1], -1*v_norm[0]])
    pt3 = np.array([center_x, center_y]) + (line_length/2)*v_perp

    return center_x, center_y, x2, y2, pt3[0], pt3[1]


def calculate_angle_between_two_line_segments(
    l1: np.ndarray,
    l2: np.ndarray,
):
    l1p1, l1p2 = l1
    l2p1, l2p2 = l2
    l1 = np.array([l1p2[0] - l1p1[0], l1p2[1] - l1p1[1]])
    l2 = np.array([l2p2[0] - l2p1[0], l2p2[1] - l2p1[1]])
    theta_rad = np.arctan2(np.cross(l1, l2), np.dot(l1, l2))
    theta_deg = np.rad2deg(theta_rad)

    if theta_deg < 0:
        return theta_deg + 360

    return theta_deg
