import itertools

import cv2 as cv
import numpy as np

from ..utils import ImageSequence


def compute_line_segment_lengths(line_segments):
    return np.linalg.norm(line_segments[:, 1, :] - line_segments[:, 0, :])


def compute_line_segments_angles(line_segments):
    """Assumes lines are expressed as (y1, x1), (y2, x2) in image coordinates"""
    line_segments_angles = np.arctan2(
        -line_segments[:, 0, 0] + line_segments[:, 1, 0],
        line_segments[:, 0, 1] - line_segments[:, 1, 1],
    )
    return np.where(
        line_segments_angles < 0, 2 * np.pi + line_segments_angles, line_segments_angles
    )


def compute_point_pairs_distances(points, point_pairs):
    return [np.linalg.norm(points[i, :] - points[j, :]) for i, j in point_pairs]


def find_closest_point_pair(points):
    point_pairs = list(itertools.combinations(list(range(0, len(points))), r=2))
    distances = compute_point_pairs_distances(points, point_pairs)
    return point_pairs[np.argmin(distances)]


def find_farthest_point_pair(points):
    point_pairs = list(itertools.combinations(list(range(0, len(points))), r=2))
    distances = compute_point_pairs_distances(points, point_pairs)
    return point_pairs[np.argmax(distances)]


def compute_median_image(image_sequence: ImageSequence, step_size=10):

    num_images = len(image_sequence)
    frames = []
    for frame_idx in np.arange(0, num_images, step_size):
        image_file_path = image_sequence[frame_idx]
        image = cv.imread(str(image_file_path), cv.IMREAD_COLOR)
        frames.append(image)
    frames = np.stack(frames, axis=3)
    median_image = np.median(frames, axis=3)
    return median_image
