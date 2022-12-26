import itertools
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import KDTree

from .utils import (
    compute_line_segments_angles,
    find_closest_point_pair,
)


class IntersectionType(Enum):
    UNKNOWN = 0
    TOP_LEFT_CORNER = 1
    TOP_RIGHT_CORNER = 2
    LEFT_GOAL_BOX_BOTTOM_CORNER = 3
    LEFT_GOAL_BOX_TOP_CORNER = 4
    RIGHT_GOAL_BOX_BOTTOM_CORNER = 5
    RIGHT_GOAL_BOX_TOP_CORNER = 6
    TOP_MIDFIELD_T_INTERSECTION = 7
    BOTTOM_MIDFIELD_T_INTERSECTION = 8
    TOP_MIDFIELD_CIRCLE_X_INTERSECTION = 9
    BOTTOM_MIDFIELD_CIRCLE_X_INTERSECTION = 10


INTERSECTIONS_3D_LOCATIONS = {
    IntersectionType.TOP_LEFT_CORNER: np.array([-4500.0, 3000.0, 0.0]),
    IntersectionType.TOP_RIGHT_CORNER: np.array([4500.0, 3000.0, 0.0]),
    IntersectionType.LEFT_GOAL_BOX_TOP_CORNER: np.array([-3900.0, 1100.0, 0]),
    IntersectionType.LEFT_GOAL_BOX_BOTTOM_CORNER: np.array([-3900.0, -1100.0, 0.0]),
    IntersectionType.RIGHT_GOAL_BOX_TOP_CORNER: np.array([3900.0, 1100.0, 0.0]),
    IntersectionType.RIGHT_GOAL_BOX_BOTTOM_CORNER: np.array([3900.0, -1100, 0.0]),
    IntersectionType.TOP_MIDFIELD_T_INTERSECTION: np.array([0, 3000, 0.0]),
    IntersectionType.BOTTOM_MIDFIELD_T_INTERSECTION: np.array([0, -3000, 0.0]),
    IntersectionType.TOP_MIDFIELD_CIRCLE_X_INTERSECTION: np.array([0, 750, 0.0]),
    IntersectionType.BOTTOM_MIDFIELD_CIRCLE_X_INTERSECTION: np.array([0, -750, 0.0]),
}


@dataclass
class IntersectionDetectionCriterion:

    angle_min: float = 0
    angle_max: float = 180
    x_min_rel: float = 0.0
    x_max_rel: float = 1.0
    y_min_rel: float = 0.0
    y_max_rel: float = 1.0
    clockwise: bool = True


INTERSECTION_DETECTION_CRITERIA = {
    IntersectionType.TOP_LEFT_CORNER: IntersectionDetectionCriterion(
        angle_min=90,
        angle_max=160,
        x_min_rel=0,
        x_max_rel=0.4,
        y_min_rel=0.0,
        y_max_rel=0.3,
        clockwise=False,
    ),
    IntersectionType.TOP_RIGHT_CORNER: IntersectionDetectionCriterion(
        angle_min=90,
        angle_max=160,
        x_min_rel=0.6,
        x_max_rel=1.0,
        y_min_rel=0.0,
        y_max_rel=0.2,
        clockwise=False,
    ),
    IntersectionType.LEFT_GOAL_BOX_TOP_CORNER: IntersectionDetectionCriterion(
        angle_min=0,
        angle_max=120,
        x_min_rel=0.0,
        x_max_rel=0.4,
        y_min_rel=0.1,
        y_max_rel=0.8,
        clockwise=False,
    ),
    IntersectionType.LEFT_GOAL_BOX_BOTTOM_CORNER: IntersectionDetectionCriterion(
        angle_min=0,
        angle_max=160,
        x_min_rel=0.0,
        x_max_rel=0.4,
        y_min_rel=0.2,
        y_max_rel=0.8,
        clockwise=True,
    ),
    IntersectionType.RIGHT_GOAL_BOX_TOP_CORNER: IntersectionDetectionCriterion(
        angle_min=0,
        angle_max=90,
        x_min_rel=0.6,
        x_max_rel=1.0,
        y_min_rel=0.2,
        y_max_rel=0.8,
        clockwise=False,
    ),
    IntersectionType.RIGHT_GOAL_BOX_BOTTOM_CORNER: IntersectionDetectionCriterion(
        angle_min=90,
        angle_max=160,
        x_min_rel=0.6,
        x_max_rel=1.0,
        y_min_rel=0.2,
        y_max_rel=0.8,
        clockwise=True,
    ),
    IntersectionType.TOP_MIDFIELD_T_INTERSECTION: IntersectionDetectionCriterion(
        angle_min=60,
        angle_max=120,
        x_min_rel=0.25,
        x_max_rel=0.75,
        y_min_rel=0.0,
        y_max_rel=0.5,
        clockwise=None,
    ),
    IntersectionType.BOTTOM_MIDFIELD_T_INTERSECTION: IntersectionDetectionCriterion(
        angle_min=60,
        angle_max=120,
        x_min_rel=0.25,
        x_max_rel=0.75,
        y_min_rel=0.5,
        y_max_rel=1.0,
        clockwise=None,
    ),
}


def find_intersecting_line_segment_pairs(line_segments):

    line_segments_angles = compute_line_segments_angles(line_segments)

    junctions, inverse = np.unique(
        line_segments.reshape(-1, 2),
        return_inverse=True,
        axis=0,
    )
    junctions_kd_tree = KDTree(junctions)
    intersecting_line_segment_pairs = set()
    for idx, p in enumerate(junctions):
        # First add all the lines that meet at this point
        lines_p = {idx2 // 2 for idx2, i in enumerate(inverse) if i == idx}
        rnn_junctions_idxs = junctions_kd_tree.query_ball_point(p, r=10)
        for junction_idx in rnn_junctions_idxs:
            lines_q = {idx2 // 2 for idx2, i in enumerate(inverse) if i == junction_idx}
            lines_p = lines_p.union(lines_q)

        lines_p = list(lines_p)

        if len(lines_p) < 2:
            continue

        line_pairs = itertools.combinations(lines_p, r=2)

        for i, j in line_pairs:
            angle_i = np.rad2deg(line_segments_angles[i]) % 180
            angle_j = np.rad2deg(line_segments_angles[j]) % 180
            if angle_i > 90:
                angle_i -= 180
            if angle_j > 90:
                angle_j -= 180

            if np.abs(angle_i - angle_j) >= 20:
                intersecting_line_segment_pairs.add((min(i, j), max(i, j)))
    return list(intersecting_line_segment_pairs)


def find_intersections(line_segments):
    intersecting_line_segment_pairs = find_intersecting_line_segment_pairs(
        line_segments
    )

    intersections = []
    for i, j in intersecting_line_segment_pairs:
        points = np.array(
            [
                line_segments[i, 0, :],
                line_segments[i, 1, :],
                line_segments[j, 0, :],
                line_segments[j, 1, :],
            ]
        )
        pair = find_closest_point_pair(points)
        p1_i, p1_j = (pair[0], pair[1]) if pair[0] < 2 else (pair[1], pair[0])
        p2_i = 0 if p1_i == 1 else 1
        p2_j = 2 if p1_j == 3 else 3

        d_i = points[p1_i, :] - points[p2_i, :]
        d_i = d_i / np.linalg.norm(d_i)
        d_j = points[p1_j, :] - points[p2_j, :]
        d_j = d_j / np.linalg.norm(d_j)

        if np.all(points[p1_i, :] == points[p1_j, :]):
            intersection_point = points[p1_i, :]
            intersections.append(np.array([intersection_point, -d_i, -d_j]))
            continue

        t = points[p2_j, :] - points[p2_i, :]
        D = np.array(
            [
                [d_i[0], -d_j[0]],
                [d_i[1], -d_j[1]],
            ]
        )

        s = np.linalg.solve(D, t)
        intersection_point = points[p2_i, :] + s[0] * d_i
        intersections.append(np.array([intersection_point, -d_i, -d_j]))
    return np.stack(intersections, axis=0)


def classify_intersection(intersection):
    intersection_point, d_i, d_j = intersection

    # Compute angle between intersection directions
    alpha = np.rad2deg(np.arccos(np.dot(d_i, d_j)))
    y0_rel, x0_rel = np.divide(intersection_point, np.array([1080, 1920]))
    y_i, x_i = intersection_point + d_i
    y_j, x_j = intersection_point + d_j

    # Compute matrix to determine intersection orientation
    clockwise_test_matrix = np.zeros((3, 3), dtype=np.float32)
    if x_i < x_j:
        clockwise_test_matrix[0, :] = np.array([x_i, 1080 - y_i, 1])
        clockwise_test_matrix[1, :] = np.array([x_j, 1080 - y_j, 1])
    else:
        clockwise_test_matrix[0, :] = np.array([x_j, 1080 - y_j, 1])
        clockwise_test_matrix[1, :] = np.array([x_i, 1080 - y_i, 1])

    clockwise_test_matrix[2, :] = np.array(
        [intersection_point[1], 1080 - intersection_point[0], 1]
    )

    is_clockwise = np.linalg.det(clockwise_test_matrix) < 0

    # Test intersection against all the rules
    for it, c in INTERSECTION_DETECTION_CRITERIA.items():
        if alpha < c.angle_min or alpha > c.angle_max:
            continue

        if x0_rel < c.x_min_rel or x0_rel > c.x_max_rel:
            continue

        if y0_rel < c.y_min_rel or y0_rel > c.y_max_rel:
            continue

        if c.clockwise is not None and is_clockwise != c.clockwise:
            continue

        return it

    return IntersectionType.UNKNOWN


def classify_intersections(intersections):
    intersection_types = [IntersectionType.UNKNOWN] * len(intersections)
    for idx, intersection in enumerate(intersections):
        intersection_type = classify_intersection(intersection)
        intersection_types[idx] = intersection_type

    return intersection_types


def detect_intersections(line_segments):

    # Find all intersections
    intersections = find_intersections(line_segments)

    # Classify all the intersections
    intersections_types = classify_intersections(intersections)

    # Assign a 3D landmark position to each intersection
    intersections_points_3d = []
    for it in intersections_types:
        if it == IntersectionType.UNKNOWN:
            intersections_points_3d.append(None)
            continue
        intersections_points_3d.append(INTERSECTIONS_3D_LOCATIONS[it])

    return intersections, intersections_types, intersections_points_3d
