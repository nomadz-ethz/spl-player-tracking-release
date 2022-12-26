import itertools
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import KDTree

import kornia.feature as KF

from .utils import compute_line_segments_angles


class KorniaSOLD2LineSegmentDetector:
    def __init__(self, config=None):
        self.device = torch.device("cpu")
        self.line_detector = KF.SOLD2_detector(
            pretrained=True,
            config=config,
        )

    def detect_line_segments(self, image) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] > 1:
            raise ValueError("Image must be sigle channel grayscale!")

        inputs = torch.tensor(
            image[None, None] / 255.0,
            dtype=torch.float,
            device=self.device,
        )

        return self.line_detector(inputs)["line_segments"][0].numpy()


class UnionFind:
    def __init__(self, ids):
        self.id_to_parent = {}
        for i in ids:
            self.id_to_parent[i] = i

    def find(self, id):
        if self.id_to_parent[id] == id:
            return id
        return self.find(self.id_to_parent[id])

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        self.id_to_parent[x] = y

    def get_disjoint_sets(self):
        res = defaultdict(list)
        for id in self.id_to_parent:
            parent_id = self.find(id)
            res[parent_id].append(id)
        return list(res.values())


def group_collinear_segments(line_segments, junctions_nn_radius=10):
    line_segments_angles = compute_line_segments_angles(line_segments)
    lines_uf = UnionFind(np.arange(line_segments.shape[0]))

    junctions, inverse = np.unique(
        line_segments.reshape(-1, 2),
        return_inverse=True,
        axis=0,
    )
    junctions_kd_tree = KDTree(junctions)

    for idx, p in enumerate(junctions):
        lines_p = {idx2 // 2 for idx2, i in enumerate(inverse) if i == idx}
        rnn_junctions_idxs = junctions_kd_tree.query_ball_point(
            p, r=junctions_nn_radius
        )
        for junction_idx in rnn_junctions_idxs:
            lines_q = {idx2 // 2 for idx2, i in enumerate(inverse) if i == junction_idx}
            lines_p = lines_p.union(lines_q)

        lines_p = list(lines_p)

        if len(lines_p) < 2:
            continue

        line_pairs = itertools.combinations(lines_p, r=2)

        merged_line_pairs = []
        for i, j in line_pairs:
            angle_i = np.rad2deg(line_segments_angles[i]) % 180
            angle_j = np.rad2deg(line_segments_angles[j]) % 180
            if angle_i > 90:
                angle_i -= 180
            if angle_j > 90:
                angle_j -= 180

            if np.abs(angle_i - angle_j) < 15:
                merged_line_pairs.append((i, j))
                lines_uf.union(i, j)

    return lines_uf.get_disjoint_sets()


def merge_collinear_line_segment_groups(
    line_segments,
    collinear_line_segments_groups,
):
    merged_line_segments = []
    for line_group in collinear_line_segments_groups:
        if len(line_group) < 2:
            merged_line_segments.append(line_segments[line_group[0], :, :])
            continue
        points = np.array([p for l in line_segments[line_group, :, :] for p in l])
        points_pairs = itertools.combinations(list(range(0, len(points))), r=2)
        farthest_pair = None
        farthest_pair_dist = 0
        for i, j in points_pairs:
            dist = np.linalg.norm(points[i] - points[j])
            if dist > farthest_pair_dist:
                farthest_pair_dist = dist
                farthest_pair = (i, j)
        i, j = farthest_pair
        first, second = (i, j) if points[i, 0] < points[j, 0] else (j, i)
        new_segment = np.array([points[first, :], points[second, :]])
        merged_line_segments.append(new_segment)
    return np.stack(merged_line_segments, axis=0)
