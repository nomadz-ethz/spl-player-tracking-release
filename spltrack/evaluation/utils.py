from typing import List
from collections import defaultdict

import numpy as np
import torch

from detectron2.structures.boxes import Boxes, pairwise_iou
from scipy.optimize import linear_sum_assignment

from ..tracking import Track


def tracks_to_mot_evaluation_data(tracks: List[Track]):
    mot_data = []
    for t in tracks:
        for frame_id, position in t.trajectory.items():
            x1, y1, x2, y2 = position
            mot_data.append(
                {
                    "frame_id": frame_id + 1,
                    "track_id": t.track_id + 1,
                    "bb_left": x1,
                    "bb_top": y1,
                    "bb_width": x2 - x1,
                    "bb_height": y2 - y1,
                    "conf": 1,
                    "x": -1,
                    "y": -1,
                    "z": -1,
                }
            )
    return sorted(mot_data, key=lambda x: (x["frame_id"], x["track_id"]))


def tracking_data_to_tracks(tracklets, tracklet_2_track_dict, player_to_track_id):
    track_id_to_player = {v: k for k, v in player_to_track_id.items()}
    tracks = {k: {} for k in player_to_track_id.keys()}
    for tracklet_id, tracklet in tracklets.items():
        if tracklet_id in tracklet_2_track_dict:
            track_id = tracklet_2_track_dict[tracklet_id]
        else:
            continue
        player_id = track_id_to_player[track_id]
        for o in tracklet:
            timestep = o["ts"]
            tracks[player_id][timestep] = tuple(o["bbox"])
    return tracks


def annotations_to_tracks(annotations, color_to_team):
    tracks = defaultdict(dict)
    for path, objects in annotations.items():
        timestep = int(path.rstrip(".jpg"))
        for o in objects:
            if o["label"] == 1:
                # this is a ball
                continue
            else:
                team = color_to_team[int(o["color"])]
                number = o["number"]
                tracks[(team, number)][timestep] = (
                    o["x_min"],
                    o["y_min"],
                    o["x_max"],
                    o["y_max"],
                )
    return dict(tracks)


def match_boxes(gt_boxes_list, pred_boxes_list):
    gt_boxes = Boxes(torch.from_numpy(np.array(gt_boxes_list)))
    pred_boxes = Boxes(torch.from_numpy(np.array(pred_boxes_list)))

    iou = pairwise_iou(gt_boxes, pred_boxes)
    match_rows, match_cols = linear_sum_assignment(iou, maximize=True)
    max_iou = iou[match_rows, match_cols]

    invalid_matches = max_iou < 0.5
    matching = match_cols[:]
    matching[invalid_matches] = -1

    return matching
