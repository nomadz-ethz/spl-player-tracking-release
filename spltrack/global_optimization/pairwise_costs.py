import numpy as np

from ..utils import index_by


def compute_tracklets_proximity_costs(
    tracking_data,
    max_tracklet_distance,
    tracklet_distance_normalizer,
    extrapolated_cost_weight=0.125,
):
    tracklets_proximity_costs = []
    tracking_data_by_id = index_by(tracking_data, "track_id")
    for tracklet1_id, tracklet1_poses in tracking_data_by_id.items():
        for tracklet2_id, tracklet2_poses in tracking_data_by_id.items():
            if tracklet2_id <= tracklet1_id:
                continue

            # If tracklets overlap in time, continue
            t_start1 = tracklet1_poses[0]["ts"]
            t_start2 = tracklet2_poses[0]["ts"]
            t_end1 = tracklet1_poses[-1]["ts"]
            t_end2 = tracklet2_poses[-1]["ts"]
            if (t_start1 <= t_end2) and (t_end1 >= t_start2):
                continue

            # Find which tracklet was earlier, which was later
            # make 1 always the first
            if t_start1 > t_start2:
                t_start1, t_start2 = t_start2, t_start1
                t_end1, t_end2 = t_end2, t_end1
                tracklet1_id, tracklet2_id = tracklet2_id, tracklet1_id
                tracklet1_poses, tracklet2_poses = tracklet2_poses, tracklet1_poses

            # Compute distance from end of earlier tracklet to start of later tracklet
            # dist in pixel coordinates
            t_dist = t_start2 - t_end1
            assert t_dist > 0
            if t_dist < max_tracklet_distance:
                end_tracklet, start_tracklet = (
                    tracklet1_poses[-1],
                    tracklet2_poses[0],
                )
                end_pose = np.array([end_tracklet["x"], end_tracklet["y"]])
                start_pose = np.array([start_tracklet["x"], start_tracklet["y"]])
                end_extrapolated = (
                    np.array([end_tracklet["vx"], end_tracklet["vy"]]) * t_dist / 1000
                )
                extrapolated_dist = np.linalg.norm(
                    end_pose + end_extrapolated - start_pose
                )
                cost = (
                    extrapolated_dist
                    / tracklet_distance_normalizer
                    * extrapolated_cost_weight
                )

                tracklets_proximity_costs.append((tracklet1_id, tracklet2_id, cost))

    return tracklets_proximity_costs
