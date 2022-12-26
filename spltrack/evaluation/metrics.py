from collections import defaultdict

import numpy as np

from ..data.constants import INVERSE_LABEL_MAP
from ..data.utils import bbox_from_points
from ..utils import index_by
from .utils import tracking_data_to_tracks, annotations_to_tracks, match_boxes

MIN_IoU = 0.5

MATCHED = 1
MISS = 2
FP = 3
DIST = 4


def bb_intersection_over_union(boxA, boxB):
    """
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def mota(
    tracking_results,
    annotations,
):
    tracking_data = tracking_results["tracking_data"]
    tracklet_to_track_id = {
        k: v for k, v in tracking_results["tracklet_to_track_id"].items() if v != -1
    }
    player_to_track_id = tracking_results["player_to_track_id"]
    color_to_team = {
        v["team_color"]: v["team_number"]
        for v in tracking_results["team_info"].values()
    }
    tracks_gt = annotations_to_tracks(annotations, color_to_team)
    tracklets = index_by(tracking_data, "track_id")
    tracks_est = tracking_data_to_tracks(
        tracklets, tracklet_to_track_id, player_to_track_id
    )
    ratios = {"Match": 0, "FP": 0, "DIST": 0, "Miss": 0}
    for player in tracks_gt.keys():
        track_gt = tracks_gt[player]
        track_est = tracks_est[player]
        common_dict = {}
        for ts, bb in track_gt.items():
            common_dict[ts] = [bb, None]
        for ts, bb in track_est.items():
            try:
                common_dict[ts][1] = bb
            except KeyError:
                common_dict[ts] = [None, bb]
        state_list = []
        keys_sorted = list(common_dict.keys())
        keys_sorted.sort()
        for k in keys_sorted:
            if common_dict[k][0] == None:
                state_list.append(FP)
            elif common_dict[k][1] == None:
                state_list.append(MISS)
            else:
                iou = bb_intersection_over_union(common_dict[k][0], common_dict[k][1])
                if iou >= MIN_IoU:
                    state_list.append(MATCHED)
                else:
                    state_list.append(DIST)
        state_list = np.array(state_list)
        ratio_match = np.mean(state_list == MATCHED)
        ratio_fp = np.mean(state_list == FP)
        ratio_dist = np.mean(state_list == DIST)
        ratio_miss = np.mean(state_list == MISS)

        ratios["Match"] += ratio_match
        ratios["FP"] += ratio_fp
        ratios["DIST"] += ratio_dist
        ratios["Miss"] += ratio_miss

    return {k: v / 10 for k, v in ratios.items()}


def mean_player_identification_accuracy(
    tracking_results,
    annotations_per_image,
    include_tns=False,
):

    fps_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)
    tps_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)
    fns_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)
    tns_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)

    player_to_track_id = tracking_results["player_to_track_id"]
    tracklet_to_track_id = tracking_results["tracklet_to_track_id"]
    color_to_team = {
        v["team_color"]: v["team_number"]
        for v in tracking_results["team_info"].values()
    }

    tracking_data_by_frame_id = index_by(tracking_results["tracking_data"], "frame_id")
    for frame_id, image in enumerate(list(sorted(annotations_per_image.keys()))):
        annotations = [
            a
            for a in annotations_per_image[image]
            if a["label"] == INVERSE_LABEL_MAP["robot"]
        ]

        tracking_data = tracking_data_by_frame_id.get(frame_id, None)
        # No tracklets on this frame -> all false negatives
        if tracking_data is None:
            fns_per_frame[frame_id] += len(annotations)
            continue

        # Get gt track boxes and player ids
        gt_tracks_boxes, gt_tracks_ids, gt_ignore = [], [], []
        for a in annotations:
            bbox = bbox_from_points(a["x_min"], a["y_min"], a["x_max"], a["y_max"])
            team_number = color_to_team[a["color"]]
            player_number = a["number"]
            ignore = a.get("ignore", False)
            track_id = player_to_track_id[(team_number, player_number)]
            gt_tracks_boxes.append(bbox)
            gt_tracks_ids.append(track_id)
            gt_ignore.append(ignore)

        # Get tracklet boxes and ids
        # TODO: should we double check that the track id is within the expected range?
        tracklet_boxes, tracklet_ids = [], []
        for t in tracking_data:
            tracklet_boxes.append(t["bbox"])
            tracklet_ids.append(t["track_id"])

        # Match based on IoU
        gt_to_pred_matching = match_boxes(gt_tracks_boxes, tracklet_boxes)
        matched_pred_tracks = []

        tp, fp, tn, fn = 0, 0, 0, 0
        for gt_track_idx, traclet_idx in enumerate(gt_to_pred_matching):
            if traclet_idx == -1 and not gt_ignore[gt_track_idx]:
                # not matched with anything
                fn += 1
                continue
            tracklet_track_id = tracklet_to_track_id[tracklet_ids[traclet_idx]]
            matched_pred_tracks.append(tracklet_ids[traclet_idx])
            if not gt_ignore[gt_track_idx]:
                if gt_tracks_ids[gt_track_idx] == tracklet_track_id:
                    tp += 1
                else:
                    fp += 1
            else:
                if tracklet_track_id == -1:
                    tn += 1 if include_tns else 0
                else:
                    fp += 1

        for pt_id in tracklet_ids:
            if pt_id in matched_pred_tracks:
                continue
            fp += 1

        tps_per_frame[frame_id] += tp
        fps_per_frame[frame_id] += fp
        tns_per_frame[frame_id] += tn
        fns_per_frame[frame_id] += fn

    total_correct = np.sum(tps_per_frame + tns_per_frame)
    total = total_correct + np.sum(fps_per_frame + fns_per_frame)
    return total_correct / total


def team_detection_precision(
    tracking_results,
    annotations_per_image,
):
    """
    Computes the accuracy of jersey detection over the whole sequence.
    """
    # Compute the predicted jersey color for each tracklet
    team_info = tracking_results["team_info"]
    color_to_team = {v["team_color"]: v["team_number"] for v in team_info.values()}
    tracking_data = tracking_results["tracking_data"]
    tracklet_id_to_pred_team_number = {}
    for tracklet_id, tracklet_data in index_by(tracking_data, "track_id").items():

        playing_teams_probs_summary = defaultdict(list)
        for tracklet_data in tracklet_data:
            playing_teams_probs = {
                ti["team_number"]: tracklet_data["team_probs"][ti["team_color"]]
                for _, ti in team_info.items()
            }
            total_team_prob_sum = sum(prob for _, prob in playing_teams_probs.items())
            for team, prob in playing_teams_probs.items():
                playing_teams_probs_summary[team].append(prob / total_team_prob_sum)

        pred_team_number = max(
            playing_teams_probs_summary,
            key=lambda k: np.mean(playing_teams_probs_summary.get(k)),
        )

        tracklet_id_to_pred_team_number[tracklet_id] = pred_team_number

    fps_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)
    tps_per_frame = np.zeros((len(annotations_per_image),), dtype=np.int64)
    fp_tracklet_ids = set()

    # Match tracklets to ground truth tracks
    tracking_data_by_frame_id = index_by(tracking_data, "frame_id")

    for frame_id, image in enumerate(list(sorted(annotations_per_image.keys()))):
        annotations = [
            a
            for a in annotations_per_image[image]
            if a["label"] == INVERSE_LABEL_MAP["robot"]
        ]

        frame_tracking_data = tracking_data_by_frame_id.get(frame_id, None)
        # No tracklets on this frame -> skip (nothing to do)
        if frame_tracking_data is None:
            continue

        # Get gt track boxes and player ids
        gt_tracks_boxes, gt_tracks_team_number, gt_ignore = [], [], []
        for a in annotations:
            bbox = bbox_from_points(a["x_min"], a["y_min"], a["x_max"], a["y_max"])
            ignore = a.get("ignore", False)
            gt_tracks_boxes.append(bbox)
            gt_tracks_team_number.append(color_to_team[a["color"]])
            gt_ignore.append(ignore)

        tracklet_boxes, tracklet_team_number = [], []
        for t in frame_tracking_data:
            tracklet_boxes.append(t["bbox"])
            tracklet_id = t["track_id"]
            pred_jersey_color = tracklet_id_to_pred_team_number[tracklet_id]
            tracklet_team_number.append(pred_jersey_color)

        gt_to_pred_matching = match_boxes(gt_tracks_boxes, tracklet_boxes)

        tp, fp = 0, 0
        for gt_track_idx, tracklet_idx in enumerate(gt_to_pred_matching):
            if tracklet_idx == -1 and not gt_ignore[gt_track_idx]:
                continue
            if not gt_ignore[gt_track_idx]:
                if (
                    gt_tracks_team_number[gt_track_idx]
                    == tracklet_team_number[tracklet_idx]
                ):
                    tp += 1
                else:
                    fp += 1

        tps_per_frame[frame_id] += tp
        fps_per_frame[frame_id] += fp

    total_correct = np.sum(tps_per_frame)
    precision = total_correct / (total_correct + np.sum(fps_per_frame))

    return precision
