from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils import index_by


def find_overlapping_events(events, target_event):
    overlapping_events = []
    for event in events:
        if (
            event["team_number"] == target_event["team_number"]
            and event["player_number"] == target_event["player_number"]
            and (
                (
                    event["start_timestamp"] >= target_event["start_timestamp"]
                    and event["start_timestamp"] <= target_event["end_timestamp"]
                )
                or (
                    event["end_timestamp"] >= target_event["start_timestamp"]
                    and event["end_timestamp"] <= target_event["end_timestamp"]
                )
                or (
                    event["start_timestamp"] <= target_event["start_timestamp"]
                    and event["end_timestamp"] >= target_event["end_timestamp"]
                )
            )
        ):
            overlapping_events.append(event)
    return overlapping_events


def clip_events(target_events, reference_events):
    clipped_target_events = []
    for te in target_events:
        overlapping_reference_events = find_overlapping_events(reference_events, te)
        if len(overlapping_reference_events) == 0:
            clipped_target_events.append(te)
        else:
            fe_start_ts = te["start_timestamp"]
            fe_end_ts = te["end_timestamp"]
            valid = True
            for re in reference_events:
                pe_start_ts = re["start_timestamp"]
                pe_end_ts = re["end_timestamp"]
                # skip if entirely inside a penalized event
                if fe_start_ts >= pe_start_ts and fe_end_ts <= pe_end_ts:
                    valid = False
                    break
                elif fe_start_ts >= pe_start_ts and fe_start_ts <= pe_end_ts:
                    te["start_timestamp"] = pe_end_ts
                elif fe_end_ts >= pe_start_ts and fe_end_ts <= pe_end_ts:
                    te["end_timestamp"] = pe_start_ts
            if valid:
                clipped_target_events.append(te)

    return clipped_target_events


def get_maybe_fallen_robot_tracking_data(tracking_data, fallen_bbox_aspect_ratio=1.5):
    # first find tracklets with aspect ratio higher than fallen_bbox_aspect_ratio
    maybe_fallen_robot_tracking_data = []
    for td in tracking_data:
        x1, y1, x2, y2 = td["bbox"]
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio > fallen_bbox_aspect_ratio:
            maybe_fallen_robot_tracking_data.append(td)
    # sort by track id and timestamp
    maybe_fallen_robot_tracking_data.sort(key=lambda x: (x["track_id"], x["frame_id"]))
    return maybe_fallen_robot_tracking_data


def get_maybe_fallen_robot_events(tracking_data, fallen_bbox_aspect_ratio=1.5):
    maybe_fallen_robot_tracking_data = get_maybe_fallen_robot_tracking_data(
        tracking_data, fallen_bbox_aspect_ratio
    )
    maybe_fallen_robot_events = []

    maybe_fallen_robot_tracking_datay_by_track_id = index_by(
        maybe_fallen_robot_tracking_data, "track_id"
    )

    for track_id, tds in maybe_fallen_robot_tracking_datay_by_track_id.items():
        # once again make sure they are sorted
        tds.sort(key=lambda x: x["frame_id"])

        # find contiguous intervals
        l = tds[0]
        r = None
        for i, td in enumerate(tds):
            if i == 0:
                continue
            if td["frame_id"] != l["frame_id"] + 1:
                if r is not None:
                    maybe_fallen_robot_events.append(
                        {
                            "start": l,
                            "end": r,
                            "track_id": track_id,
                        }
                    )
                l = td
                r = None
            else:
                r = td

            if i == len(tds) - 1 and r is not None:
                maybe_fallen_robot_events.append(
                    {
                        "start": l,
                        "end": r,
                        "track_id": track_id,
                    }
                )

    return maybe_fallen_robot_events


def match_tracklets_to_fallen_events(
    tracking_data,
    fallen_events,
    fallen_bbox_aspect_ratio: float = 1.5,
):
    maybe_fallen_robot_events = get_maybe_fallen_robot_events(
        tracking_data, fallen_bbox_aspect_ratio
    )

    maybe_fallen_robot_track_ids = list(
        set([mfe["track_id"] for mfe in maybe_fallen_robot_events])
    )
    maybe_fallen_robot_track_ids.sort()

    # build cost matrix
    utility_matrix = np.zeros((len(maybe_fallen_robot_track_ids), len(fallen_events)))

    for i, mfe in enumerate(maybe_fallen_robot_events):
        for j, fe in enumerate(fallen_events):
            mft_start_ts = mfe["start"]["ts"]
            mft_end_ts = mfe["end"]["ts"]
            fe_start_ts = fe["start_timestamp"]
            fe_end_ts = fe["end_timestamp"]
            d = min(mft_end_ts, fe_end_ts) - max(mft_start_ts, fe_start_ts)
            if d > 0:
                utility = d / (mft_end_ts - mft_start_ts)
                k = maybe_fallen_robot_track_ids.index(mfe["track_id"])
                utility_matrix[k, j] += utility

    # solve assignment problem
    track_id_to_fallen_events = defaultdict(list)
    row_ind, col_ind = linear_sum_assignment(utility_matrix, maximize=True)
    for i, j in zip(row_ind, col_ind):
        track_id = maybe_fallen_robot_track_ids[i]
        fe = fallen_events[j]
        track_id_to_fallen_events[track_id].append(fe)

    return track_id_to_fallen_events
