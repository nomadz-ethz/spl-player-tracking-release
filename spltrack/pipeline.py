import logging
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torchvision.transforms.functional as TF
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .team_detection import TeamDetector
from .number_recognition import NumberDetector
from .tracking import Tracktor
from .global_optimization import GlobalOptimizer
from .global_optimization.unary_costs import (
    compute_self_localization_costs,
    compute_team_detection_costs,
    compute_tracklet_lifetime_costs,
    compute_fallen_events_costs,
)
from .global_optimization.pairwise_costs import (
    compute_tracklets_proximity_costs,
)
from .utils import (
    VisualTrackingData,
    Camera,
    ImageSequence,
    PoseKalmanFilter,
    save_tracks,
    load_tracks,
    index_by,
)
from .game_data import (
    TeamComm,
    GameState,
    clip_events,
    match_tracklets_to_fallen_events,
)


def _compute_localization_confidences(
    poses, vmax=1.5, penalty_ramp=10, penalty_duration=5
):
    # Compute the confidence for each pose in a sequence of poses
    # poses: list of dicts with at least keys "ts", "x", "y"
    # vmax (m/s): any speed above this is penalized (& a few poses before and after too)
    # penalty_ramp: how many frames the confidence ramps down to 0 & how many frames to ramp back up
    # penalty_duration: how many frames the confidence stays at 0
    confidences = np.ones(len(poses))
    impulses = np.zeros(len(poses))
    for i, pose in enumerate(poses):
        if (pose["x"], pose["y"]) == (0.0, 0.0):
            impulses[i] = 1.0
    for i, (a, b) in enumerate(zip(poses[:-1], poses[1:])):
        t, x, y = b["ts"], b["x"], b["y"]
        dt = b["ts"] - a["ts"]
        if (x, y) == (0.0, 0.0):
            impulses[i] = 1.0
            continue
        if dt != 0.0:
            v = np.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) / (
                b["ts"] - a["ts"]
            )
            if v < vmax:
                impulses[i] = 1.0
            # print(np.quantile(np.array(vels), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    # TODO Kernel might need to depend on the timestamps instead of being linear in frame index?
    kernel = np.concatenate(
        [
            np.linspace(-1.0, 0.0, num=penalty_ramp - 1, endpoint=False),
            np.zeros(max(0, penalty_duration - 1)),
            np.linspace(0.0, -1.0, num=penalty_ramp),
        ]
    )
    smooth_impulses = np.convolve(impulses, kernel, "same")[: len(impulses)]
    smooth_impulses = np.maximum(smooth_impulses, -1.0)
    confidences = confidences - smooth_impulses
    return confidences


def _unify_costs(cost_triplets: List[Tuple[int, int, float]], offset=0):
    pair_to_costs = defaultdict(list)
    for p1, p2, cost in cost_triplets:
        pair_to_costs[(p1, p2)].append(cost)

    new_cost_triplets = []
    for (p1, p2), costs in pair_to_costs.items():
        new_cost_triplets.append((p1, p2, np.sum(costs) + offset))

    return new_cost_triplets


def _extract_costs(file, cost_to_extract):
    if not file.is_file():
        return None
    with open(file, "rb") as f:
        data = np.load(f)
        return data.get(cost_to_extract, None)


class TrackingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self._tracker = None
        self._team_detector = None
        self._number_detector = None
        self._pose_filter = PoseKalmanFilter()
        self._optimizer = GlobalOptimizer.from_config(self.cfg)
        logging.info(self.cfg.dump())

    def _predict_tracklets(self, image_sequence: ImageSequence):
        if self._tracker is None:
            self._tracker = Tracktor.from_config(self.cfg)
        self._tracker.reset()
        for frame_id, image_file_path in enumerate(tqdm(image_sequence)):
            image = cv.imread(str(image_file_path))
            _ = self._tracker.track(frame_id, image)
        return self._tracker.get_all_tracks()

    def _detect_teams(self, image_sequence, tracking_data):
        if self._team_detector is None:
            self._team_detector = TeamDetector.from_config(self.cfg)
        team_dets = []
        for ts, tds in tqdm(index_by(tracking_data, "ts").items()):
            if len(tds) == 0:
                continue
            frame_id = image_sequence.get_frame_id(ts)
            if frame_id is None:
                continue
            bboxes = [x["bbox"].astype(int) for x in tds]
            image = Image.fromarray(cv.imread(str(image_sequence[frame_id])))
            cropped_images = [
                TF.crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
                for bbox in bboxes
            ]
            preds = self._team_detector(cropped_images)
            assert len(preds) == len(tds)
            for ts_idx, team_probs in enumerate(preds):
                tds[ts_idx]["team_probs"] = team_probs
                team_dets.append(
                    {
                        "frame_id": frame_id,
                        "track_id": tds[ts_idx]["track_id"],
                        "team_probs": team_probs,
                    }
                )
        return team_dets

    def _detect_numbers(self, image_sequence, tracking_data):
        if self._number_detector is None:
            self._number_detector = NumberDetector.from_config(self.cfg)
        number_dets = []

        for ts, tds in tqdm(index_by(tracking_data, "ts").items()):
            if len(tds) == 0:
                continue
            frame_id = image_sequence.get_frame_id(ts)
            if frame_id is None:
                continue
            bboxes = [x["bbox"].astype(int) for x in tds]
            image = Image.open(image_sequence[frame_id])
            cropped_images = [
                TF.crop(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
                for bbox in bboxes
            ]
            preds = self._number_detector(cropped_images)
            assert len(preds) == len(tds)
            for td_idx, number in enumerate(preds):
                tds[td_idx]["number"] = number
                number_dets.append(
                    {
                        "frame_id": frame_id,
                        "track_id": tds[td_idx]["track_id"],
                        "number": number,
                    }
                )
        return number_dets

    def _match_tracklets_to_fallen_events(
        self, tracking_data, game_state: GameState, team_comm: TeamComm
    ):
        fallen_events = clip_events(
            target_events=team_comm.get_fallen_events(),
            reference_events=game_state.get_penalized_events(),
        )
        return match_tracklets_to_fallen_events(
            tracking_data,
            fallen_events,
            fallen_bbox_aspect_ratio=2,
        )

    def _get_incompatible_tracklet_player_pairs(
        self, tracking_data, game_state: GameState
    ):
        """
        Get pairs of tracklets and players that are incompatible with each other
        because the player is penalized.
        """
        penalized_events = game_state.get_penalized_events()
        tracking_data_by_track_id = index_by(tracking_data, "track_id")
        incompatible_pairs = []
        for pe in penalized_events:
            pe_start_ts = pe["start_timestamp"]
            pe_end_ts = pe["end_timestamp"]
            invalid_start_ts = pe_start_ts + self.cfg.MISC.PENALIZED_EVENT_DELAY
            invalid_end_ts = pe_end_ts - self.cfg.MISC.PENALIZED_EVENT_DELAY
            for track_id, tds in tracking_data_by_track_id.items():
                start_ts = min([td["ts"] for td in tds])
                end_ts = max([td["ts"] for td in tds])
                # check if tracklet overlaps with penalized event
                if (
                    invalid_start_ts <= start_ts <= invalid_end_ts
                    and invalid_start_ts <= end_ts <= invalid_end_ts
                ):
                    incompatible_pairs.append(
                        (track_id, (pe["team_number"], pe["player_number"]))
                    )

        return incompatible_pairs

    def _get_temporally_overlapping_tracklet_pairs(self, tracklets):
        patience = self.cfg.TRACKER.INACTIVE_PATIENCE
        t_start = np.array([t.start_frame_id for t in tracklets])
        t_end = np.array([t.last_frame_id - patience for t in tracklets])
        overlap_matrix = (t_start[:, np.newaxis] <= t_end[np.newaxis, :]) & (
            t_end[:, np.newaxis] >= t_start[np.newaxis, :]
        )
        # remove elements in upper triangle + diagonal
        np.fill_diagonal(overlap_matrix, False)
        overlap_matrix *= np.tri(*overlap_matrix.shape, dtype=bool)
        overlap_indices = np.where(overlap_matrix)
        overlap_indices = [(i, j) for (i, j) in zip(*overlap_indices)]
        return overlap_indices

    def run(self, data_dir_path: Path, output_dir_path: Path = None):

        cached_tracking_data_dir = data_dir_path / "cached_tracking_data"
        cached_tracking_data_dir.mkdir(exist_ok=True)
        if output_dir_path is None:
            output_dir_path = cached_tracking_data_dir
        output_dir_path.mkdir(exist_ok=True)
        numpy_cached_file = output_dir_path / "optimization_data.npz"

        logging.info("Loading camera parameters.")

        # Load camera intrinsics
        dist_coeffs_file_path = data_dir_path / "dist_coeffs.txt"
        if not dist_coeffs_file_path.is_file():
            raise ValueError(f"Camera distortion coeffs file not found!")
        dist_coeffs = np.loadtxt(str(dist_coeffs_file_path))
        camera_matrix_file_path = data_dir_path / "camera_matrix.txt"
        if not camera_matrix_file_path.is_file():
            raise ValueError(f"Camera matrix file not found!")
        camera_matrix = np.loadtxt(str(camera_matrix_file_path))

        # Load camera extrinsics
        try:
            camera_extrinsics_file_path = next(data_dir_path.glob("*extrinsics.npz"))
            with np.load(str(camera_extrinsics_file_path)) as data:
                rvec = data["rvec"]
                tvec = data["tvec"]
        except StopIteration:
            logging.info("Camera extrinsics not found. Starting calibration.")
            raise NotImplementedError(
                "automatic camera calibration not integrated yet!"
            )

        # Build camera
        camera = Camera(camera_matrix, dist_coeffs, rvec, tvec)

        logging.info("Loading image sequence.")
        # Load image sequence
        image_sequence = ImageSequence(data_dir_path / "images")

        gc_dir_path = data_dir_path / "gc/json"

        logging.info("Loading game data.")

        # Load gc logs
        try:
            g_logs_file_path = next(gc_dir_path.glob("*.log.g.json"))
        except StopIteration:
            raise FileNotFoundError("Team comm logs not found!")

        game_state = GameState.read_from_json(g_logs_file_path)

        # Get home and away team number and jersey color
        team_info = game_state.get_team_info()

        # Load team comm
        try:
            team_comm_logs_file_path = next(
                data_dir_path.joinpath("gc/json").glob("*.log.tc.json")
            )
        except StopIteration:
            raise FileNotFoundError("Team comm logs not found!")

        team_comm = TeamComm.read_from_json(team_comm_logs_file_path)

        # Extract self loc poses from team comm data
        team_comm_poses = []
        for msg in team_comm:
            ts, team, player_num, pose = (
                msg["timestamp"],
                msg["teamNum"],
                msg["playerNum"],
                msg["pose"],
            )
            # ignore players from other teams which happen to be in the game controller
            if (
                team != team_info["home"]["team_number"]
                and team != team_info["away"]["team_number"]
            ):
                continue
            # FIXME: fix substitue edge case
            if player_num > 5:
                continue
            x, y, _ = pose  # pose is x, y, r but we don't use r
            if team != team_info["home"]["team_number"]:
                x, y = -x, -y
            ix, iy = camera.project_field_points_on_image(np.array([x, y, 0.0]))[0][
                0, 0
            ]
            team_comm_poses.append(
                {
                    "ts": ts,
                    "player": (team, player_num),
                    "x": x,
                    "y": y,
                    "ix": ix,
                    "iy": iy,
                    "confidence": float("nan"),  # to be calculated later
                }
            )

        team_comm_poses_by_player = index_by(team_comm_poses, "player")

        # Compute confidence in a sequence of team comm poses &
        for player, poses in team_comm_poses_by_player.items():
            confidences = _compute_localization_confidences(poses)
            for pose, confidence in zip(poses, confidences):
                pose["confidence"] = confidence

        players = sorted(
            team_comm_poses_by_player.keys(),
            key=lambda player: (-1, player[1])
            if player[0] == team_info["home"]["team_number"]
            else player,
        )
        player_to_track_id = {
            player: track_id for track_id, player in enumerate(players)
        }
        track_id_to_player = {v: k for k, v in player_to_track_id.items()}

        # Load cached tracklets or predict with Tracktor
        tracklets_file_path = cached_tracking_data_dir / "tracklets.csv"
        if tracklets_file_path.is_file():
            logging.info("Loading cached tracklets.")
            tracklets = load_tracks(tracklets_file_path)
        else:
            logging.info("Predictin tracklets with Tracktor.")
            tracklets = self._predict_tracklets(image_sequence)
            # Save tracklets to file
            logging.info(f"Saving tracklets to {str(tracklets_file_path)}")
            save_tracks(tracklets_file_path, tracklets)

        # Convert tracklets to flat representation
        tracking_data = VisualTrackingData.from_tracklets(tracklets)
        for td in tracking_data:
            td["ts"] = image_sequence.get_frame_timestamp(td["frame_id"])
            bbox = td["bbox"]
            td["ix"], td["iy"] = (bbox[0] + bbox[2]) / 2, bbox[3]

        # Compute tracklets position in field coordinates
        for td, field_position in zip(
            tracking_data,
            camera.project_image_points_on_field(
                np.array([[t["ix"], t["iy"]] for t in tracking_data])
            ),
        ):
            td["x_raw"], td["y_raw"] = field_position

        number_dets_file_path = cached_tracking_data_dir / "number_dets.csv"
        if number_dets_file_path.is_file():
            logging.info("Loading cached number detections.")
            number_dets = pd.read_csv(number_dets_file_path).to_dict("records")
            tracking_data.add_number_detections(number_dets)
        else:
            logging.info("Detecting numbers")
            number_dets = self._detect_numbers(image_sequence, tracking_data)
            number_dets.sort(key=lambda x: (x["frame_id"], x["track_id"]))
            pd.DataFrame(number_dets).to_csv(
                number_dets_file_path, index=False, header=True
            )

        team_dets_file_path = cached_tracking_data_dir / "team_dets.pkl"
        if team_dets_file_path.is_file():
            logging.info("Loading cached team detections.")
            team_dets = pd.read_pickle(team_dets_file_path).to_dict("records")
            tracking_data.add_team_detections(team_dets)
        else:
            logging.info("Detecting team")
            team_dets = self._detect_teams(image_sequence, tracking_data)
            pd.DataFrame(team_dets).to_pickle(team_dets_file_path)

        # Smooth tracklet positions in field coordinates
        for _, tracklet_tracking_data in index_by(tracking_data, "track_id").items():
            self._pose_filter.filter(tracklet_tracking_data)

        # Detect fallen events
        track_id_to_fallen_events = self._match_tracklets_to_fallen_events(
            tracking_data, game_state, team_comm
        )

        tracklet_to_track_costs = []
        tracklet_to_track_cost_details = defaultdict(dict)
        self_loc_costs = _extract_costs(numpy_cached_file, "localization_costs")
        if self_loc_costs is None:
            self_loc_costs = compute_self_localization_costs(
                tracking_data,
                team_comm_poses,
                player_to_track_id,
                max_distance=self.cfg.COST_COMPUTER.MAX_SELF_LOCALIZATION_MATCHING_DISTANCE,
            )
        for tracklet_id, track_id, cost in self_loc_costs:
            weighted_cost = self.cfg.COST_COMPUTER.SELF_LOCALIZATION_COST_WEIGHT * cost
            tracklet_to_track_costs.append((tracklet_id, track_id, weighted_cost))
            tracklet_to_track_cost_details[tracklet_id, track_id]["loc"] = weighted_cost

        team_detection_costs = _extract_costs(numpy_cached_file, "team_detection_costs")
        if team_detection_costs is None:
            team_detection_costs = compute_team_detection_costs(
                tracking_data, team_info, track_id_to_player
            )
        for tracklet_id, track_id, cost in team_detection_costs:
            weighted_cost = self.cfg.COST_COMPUTER.TEAM_DETECTION_COST_WEIGHT * cost
            tracklet_to_track_costs.append((tracklet_id, track_id, weighted_cost))
            tracklet_to_track_cost_details[tracklet_id, track_id][
                "team"
            ] = weighted_cost
        tracklet_lifetime_costs = _extract_costs(
            numpy_cached_file, "tracklet_lifetime_costs"
        )
        if tracklet_lifetime_costs is None:
            tracklet_lifetime_costs = compute_tracklet_lifetime_costs(
                tracking_data=tracking_data,
                num_players=len(players),
                min_tracklet_lifetime=self.cfg.COST_COMPUTER.MIN_TRACKLET_LIFETIME,
            )
        for tracklet_id, track_id, cost in tracklet_lifetime_costs:
            weighted_cost = self.cfg.COST_COMPUTER.TRACKLET_LIFETIME_COST_WEIGHT * cost

            tracklet_to_track_costs.append((tracklet_id, track_id, weighted_cost))
            tracklet_to_track_cost_details[tracklet_id, track_id]["len"] = weighted_cost

        fallen_events_costs = _extract_costs(numpy_cached_file, "fallen_events_costs")
        if fallen_events_costs is None:
            fallen_events_costs = compute_fallen_events_costs(
                tracklet_to_fallen_events=track_id_to_fallen_events,
                player_to_track_id=player_to_track_id,
                num_players=len(players),
            )
        for tracklet_id, track_id, cost in fallen_events_costs:
            weighted_cost = self.cfg.COST_COMPUTER.FALLEN_EVENTS_COST_WEIGHT * cost
            tracklet_to_track_costs.append((tracklet_id, track_id, weighted_cost))

        # Compute pairwise costs
        tracklet_to_tracklet_costs = []
        tracklet_proximity_costs = _extract_costs(
            numpy_cached_file, "tracklet_proximity_costs"
        )
        if tracklet_proximity_costs is None:
            tracklet_proximity_costs = compute_tracklets_proximity_costs(
                tracking_data=tracking_data,
                max_tracklet_distance=self.cfg.COST_COMPUTER.MAX_PAIRWISE_TRACKLET_DISTANCE,
                tracklet_distance_normalizer=self.cfg.COST_COMPUTER.PAIRWISE_DISTANCE_COST_NORMALIZER,
                extrapolated_cost_weight=1 / 8,
            )
        for tracklet1_id, tracklet2_id, cost in tracklet_proximity_costs:
            cost = self.cfg.COST_COMPUTER.TRACKLET_TO_TRACKLET_COST_WEIGHT * cost
            tracklet_to_tracklet_costs.append((tracklet1_id, tracklet2_id, cost))

        # Unify the costs
        tracklet_to_track_costs = _unify_costs(
            tracklet_to_track_costs,
            offset=self.cfg.COST_COMPUTER.UNARY_COSTS_OFFSET,
        )
        tracklet_to_tracklet_costs = _unify_costs(
            tracklet_to_tracklet_costs,
            offset=self.cfg.COST_COMPUTER.PAIRWISE_COSTS_OFFSET,
        )

        # find incompatible pairs
        incompatible_tracklet_track_pairs = []

        if self.cfg.MISC.IGNORE_PENALIZED_PLAYERS:
            incompatible_tracklet_track_pairs += [
                (p[0], player_to_track_id[p[1]])
                for p in self._get_incompatible_tracklet_player_pairs(
                    tracking_data, game_state
                )
            ]

        # find incompatible tracklet pairs
        incompatible_tracklet_tracklet_pairs = _extract_costs(
            numpy_cached_file, "temporal_overlapping_pairs"
        )
        if incompatible_tracklet_tracklet_pairs is None:
            incompatible_tracklet_tracklet_pairs = []
            incompatible_tracklet_tracklet_pairs += (
                self._get_temporally_overlapping_tracklet_pairs(tracklets)
            )

        # make sure there are no duplicates
        incompatible_tracklet_track_pairs = list(set(incompatible_tracklet_track_pairs))
        incompatible_tracklet_tracklet_pairs = list(
            set(incompatible_tracklet_tracklet_pairs)
        )

        # Optimize tracklets
        tracklet_to_track_id = self._optimizer.optimize(
            tracklets,
            tracklet_to_track_costs,
            tracklet_to_tracklet_costs,
            incompatible_tracklet_track_pairs,
            incompatible_tracklet_tracklet_pairs,
        )

        # Map tracklets to tracks
        matched_tracklets = []
        excluded_tracklets = []
        for tracklet in deepcopy(tracklets):
            track_id = tracklet_to_track_id[tracklet.track_id]
            if track_id == -1:
                excluded_tracklets.append(tracklet)
            else:
                tracklet.track_id = track_id
                matched_tracklets.append(tracklet)

        # Save the result
        save_tracks(
            output_dir_path / "tracks.csv",
            matched_tracklets,
        )

        save_tracks(
            output_dir_path / "excluded_tracklets.csv",
            excluded_tracklets,
        )
        if self.cfg.COST_COMPUTER.SAVE_COSTS:
            np.savez(
                str(output_dir_path / "optimization_data.npz"),
                localization_costs=np.array(self_loc_costs),
                team_detection_costs=np.array(team_detection_costs),
                tracklet_lifetime_costs=np.array(tracklet_lifetime_costs),
                tracklet_proximity_costs=np.array(tracklet_proximity_costs),
                fallen_events_costs=np.array(fallen_events_costs),
                temporal_overlapping_pairs=np.array(
                    incompatible_tracklet_tracklet_pairs
                ),
                tracklets_to_track_assignment=np.array(
                    list(tracklet_to_track_id.items())
                ),
            )

        return {
            "camera": camera,
            "tracklets": tracklets,
            "team_comm_poses_by_ts": index_by(team_comm_poses, "ts"),
            "tracklet_to_track_cost_details": tracklet_to_track_cost_details,
            "tracklet_to_tracklet_costs": tracklet_to_tracklet_costs,
            "tracks": matched_tracklets,
            "excluded_tracklets": excluded_tracklets,
            "tracking_data": tracking_data,
            "tracklet_to_track_id": tracklet_to_track_id,
            "player_to_track_id": player_to_track_id,
            "track_id_to_player": track_id_to_player,
            "team_info": team_info,
            "tracklet_to_fallen_events": track_id_to_fallen_events,
        }
