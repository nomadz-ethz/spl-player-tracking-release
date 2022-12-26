from collections import defaultdict

import numpy as np

from ..utils import index_by


def _interpolate_pose(request_ts, poses_by_ts):
    # ts: (...,)
    # poses_by_ts: (:, >=3), with cols [ts, x, y, *others]
    # return: (..., >=2), with cols [x, y, *others]
    if not poses_by_ts.shape[0]:
        raise ValueError("Nothing to interpolate")
    return np.stack(
        [
            np.interp(request_ts, poses_by_ts[:, 0], poses_by_ts[:, col])
            for col in range(1, poses_by_ts.shape[1])
        ],
        axis=-1,
    )


def _mean_normalized_pose_distance_meters(
    poses, reference_poses_by_ts, max_distance=None
):
    # Average of the distance between each pose in `poses` and the time-interpolated reference pose, weighted by pose confidence
    # poses: List[Dict{ts, x, y}]
    # reference_poses_by_ts: (:, 3), with cols [ts, x, y, confidence]
    pose_tss = [pose["ts"] for pose in poses]
    pose_xys = np.array([[pose["x"], pose["y"]] for pose in poses])  # (:, 2)
    reference_poses = _interpolate_pose(pose_tss, reference_poses_by_ts)  # (:, 2)
    distances = np.sqrt(((pose_xys - reference_poses[:, :2]) ** 2).sum(axis=1))
    weights = reference_poses[:, 2]
    weighted_distances = weights * distances
    total_weight = np.sum(weights)
    if total_weight == 0.0:
        # HACK Big number just to say "don't ever try to match this"
        return 99999.0
    return np.sum(weighted_distances) / total_weight


def compute_self_localization_costs(
    tracking_data,
    team_comm_poses,
    player_to_track_id,
    max_distance,
):
    self_localization_costs = []

    team_comm_poses_by_ts_by_player = dict()
    for player, poses in index_by(team_comm_poses, "player").items():
        poses_by_ts = index_by(poses, "ts")
        sorted_ts = sorted(pose["ts"] for pose in poses)
        # Assumes len(poses_by_ts[ts]) == 1 for all ts
        team_comm_poses_by_ts_by_player[player] = np.array(
            [
                [
                    ts,
                    poses_by_ts[ts][0]["x"],
                    poses_by_ts[ts][0]["y"],
                    poses_by_ts[ts][0]["confidence"],
                ]
                for ts in sorted_ts
            ]
        )

    for tracklet_id, tracklet_poses in index_by(tracking_data, "track_id").items():
        for player, player_poses_by_ts in team_comm_poses_by_ts_by_player.items():
            track_id = player_to_track_id[player]

            # Compute (player_poses, tracklet_poses) distance
            distance = _mean_normalized_pose_distance_meters(
                tracklet_poses,
                player_poses_by_ts,
                max_distance,
            )
            # Scale to a cost value
            cost = distance / max_distance

            self_localization_costs.append((tracklet_id, track_id, cost))
    return self_localization_costs


def compute_team_detection_costs(tracking_data, team_info, track_id_to_player):
    team_detection_costs = []
    # Prefer matches where team color detection matches track team
    for tracklet_id, tracklet_data in index_by(tracking_data, "track_id").items():
        # Summarize all team probabilities
        # Dict[team, List[probs]]
        playing_teams_probs_summary = defaultdict(list)
        for td in tracklet_data:
            playing_teams_probs = {
                ti["team_number"]: td["team_probs"][ti["team_color"]]
                for _, ti in team_info.items()
            }
            total_team_prob_sum = sum(
                prob for team, prob in playing_teams_probs.items()
            )
            for team, prob in playing_teams_probs.items():
                playing_teams_probs_summary[team].append(prob / total_team_prob_sum)

        # Compute cost value: higher mean team prob (0..1) -> lower cost (1..0)
        for track_id, (team, player_num) in track_id_to_player.items():
            prob = np.mean(playing_teams_probs_summary[team])
            cost = 1 - prob
            team_detection_costs.append((tracklet_id, track_id, cost))
    return team_detection_costs


def compute_tracklet_lifetime_costs(tracking_data, num_players, min_tracklet_lifetime):
    tracklet_lifetime_costs = []
    # Prefer matching longer tracklets
    for tracklet_id, tracklet_poses in index_by(tracking_data, "track_id").items():
        tracklet_poses_ts = [pose["ts"] for pose in tracklet_poses]
        tracklet_lifetime = max(tracklet_poses_ts) - min(tracklet_poses_ts)
        for track_id in range(num_players):
            # fmt: off
            cost = 1 / (tracklet_lifetime / min_tracklet_lifetime) * 4
            cost = min(1, cost)
            tracklet_lifetime_costs.append((tracklet_id, track_id, cost))
            # fmt: on
    return tracklet_lifetime_costs


def compute_fallen_events_costs(
    tracklet_to_fallen_events, player_to_track_id, num_players
):
    fallen_events_costs = []
    for tracklet_id, fallen_events in tracklet_to_fallen_events.items():
        for fe in fallen_events:
            player_id = fe["team_number"], fe["player_number"]
            track_id = player_to_track_id[player_id]
            # penalize matching this tracklet to other tracks
            for id_track in range(num_players):
                if track_id != id_track:
                    fallen_events_costs.append(
                        (tracklet_id, id_track, 1)
                    )  # Not sure. But this is a bug for sure
    return fallen_events_costs
