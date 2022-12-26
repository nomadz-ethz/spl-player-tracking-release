from pathlib import Path
from typing import List
from collections import defaultdict

import cv2 as cv
import numpy as np
import pandas as pd

from .tracking import Track
from .evaluation.utils import tracks_to_mot_evaluation_data


class VisualTrackingData:
    def __init__(self, tracking_data):
        self._tracking_data = tracking_data

        # preindex the data for fast lookups
        self._tracking_data_by_frame_and_track_id = {
            (t["frame_id"], t["track_id"]): t for t in self._tracking_data
        }

    def data(self):
        """Returns the underlying tracking data as a list of dicts."""
        return self._tracking_data

    @classmethod
    def from_tracklets(cls, tracklets):
        return cls(flatten_tracks(tracklets))

    def __getitem__(self, idx):
        return self._tracking_data[idx]

    def __len__(self):
        return len(self._tracking_data)

    def add_number_detections(self, number_dets):
        for d in number_dets:
            td = self._tracking_data_by_frame_and_track_id.get(
                (d["frame_id"], d["track_id"]), None
            )
            if td is None:
                raise ValueError("Tracklets and number detection do not match!")
            td["number"] = d["number"]

    def add_team_detections(self, team_dets):
        for d in team_dets:
            td = self._tracking_data_by_frame_and_track_id.get(
                (d["frame_id"], d["track_id"]), None
            )
            if td is None:
                raise ValueError("Tracklets and team detection do not match!")
            td["team_probs"] = d["team_probs"]


class Camera:
    def __init__(
        self,
        camera_mat: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ):
        self.camera_mat = camera_mat
        self.camera_mat_inv = np.linalg.inv(self.camera_mat)
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
        self.rmat = cv.Rodrigues(rvec)[0]
        self.rmat_inv = np.linalg.inv(self.rmat)
        self.camera_pos = -self.rmat_inv @ self.tvec[:, 0]

    def undistort_points(self, image_points: np.ndarray):
        distortion_center = self.camera_mat[0:2, 2]
        r = np.linalg.norm(image_points - distortion_center, axis=-1)
        return (
            distortion_center
            + (image_points - distortion_center)
            * np.polyval(self.dist_coeffs[::-1] + [0], r)[..., None]
        )

    def project_image_points_on_field(self, image_points):

        undistorted_image_points = self.undistort_points(image_points)

        *extra_dims, _ = undistorted_image_points.shape

        # (..., 2) -> (..., 3, 1)
        image_points_homogeneous = np.concatenate(
            [image_points, np.ones(list(extra_dims) + [1])], axis=-1
        )[..., None]
        camera_rays = self.camera_mat_inv @ image_points_homogeneous  # (..., 3, 1)
        world_rays = self.rmat_inv @ camera_rays  # (..., 3, 1)
        ray_lens = self.camera_pos[-1] / world_rays[..., -1, 0]  # (...,)

        world_points_on_field = (
            self.camera_pos - world_rays[..., 0] * ray_lens[..., None]
        )
        return world_points_on_field[..., :2]

    def project_field_points_on_image(self, points: np.ndarray) -> np.ndarray:
        """
        Projects points from 3D to the camera plane
        """
        return cv.projectPoints(
            points,
            self.rvec,
            self.tvec,
            self.camera_mat,
            np.array([]),
        )


class ImageSequence:
    def __init__(self, images_dir_path: Path):

        self.images_dir_path = images_dir_path
        self._images = sorted(self.images_dir_path.glob("*.jpg"))
        if len(self._images) == 0:
            print(f"Warning: no images found in {str(images_dir_path)}")
        self._timestamp_to_frame_id = {
            int(p.stem): frame_id for frame_id, p in enumerate(self._images)
        }

    def __getitem__(self, idx):
        return self._images[idx]

    def __len__(self):
        return len(self._images)

    def get_frame_timestamp(self, frame_idx):
        if frame_idx < 0 or frame_idx >= len(self._images):
            return None
        return int(self._images[frame_idx].stem)

    def get_frame_id(self, timestamp):
        return self._timestamp_to_frame_id.get(timestamp, None)


class PoseKalmanFilter:
    """
    Use Kalman filter with state (x, y, vx, vy) & observations (x_raw, y_raw)
    Process model is (x, y, vx, vy)[k+1] = (x + vx, y + vy, vx, vy)[k] + N(0, diag([0 0 vx_noise vy_noise]))
    or s[k+1] = [[1 0 1 0] [0 1 0 1] [0 0 1 0] [0 0 0 1]] @ s[k] + N(0, diag([0 0 vx_noise vy_noise]))
    Observation model is (x_raw, y_raw)[k] = (x, y)[k] + N(0, diag([x_raw_noise y_raw_noise]))
    or o[k] = [[1 0 0 0] [0 1 0 0]] s[k] + N(0, diag([x_raw_noise y_raw_noise]))
    """

    def __init__(self, q=500, p=1000):
        self.Q0 = np.diag(
            [0, 0, q**2, q**2]
        )  # Process noise covariance (if in 1 second steps)
        self.R = np.diag([q, q]) ** 2  # Observation noise covariance
        self.P0 = np.diag([q, q, p, p]) ** 2  # Initial state covariance

    def filter(self, poses):
        poses = sorted(
            poses, key=lambda pose: pose["ts"]
        )  # make sure poses are sorted by timestamp
        pose = poses[0]
        # Initial state (x [mm], y [mm], vx [mm/s], vy [mm/s])
        vx0, vy0 = 0, 0
        t0, s0 = pose["ts"], np.array([pose["x_raw"], pose["y_raw"], vx0, vy0])
        pose["x"], pose["y"], pose["vx"], pose["vy"] = s0

        # Feed in observed poses
        t, s, P = t0, s0, self.P0
        for pose in poses[1:]:
            t2 = pose["ts"]
            dt = (t2 - t) / 1000  # dt [s]
            # Kalman filter params
            F = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
            )  # Process matrix
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
            Q = self.Q0 * dt  # Process noise covariance

            # Predict
            s1 = F @ s
            P1 = F @ P @ F.T + Q
            # Update
            x_obs, y_obs = pose["x_raw"], pose["y_raw"]
            innovation = np.array([x_obs, y_obs]) - H @ s1
            innovation_cov = H @ P1 @ H.T + self.R
            K = P1 @ H.T @ np.linalg.inv(innovation_cov)
            s2 = s1 + K @ innovation
            P2 = (np.eye(4) - K @ H) @ P1

            # Iterate
            pose["x"], pose["y"], pose["vx"], pose["vy"] = s2
            t, s, P = t2, s2, P2


def save_tracks(tracks_file_path: Path, tracks: List[Track]):
    tracks_df = pd.DataFrame(tracks_to_mot_evaluation_data(tracks))
    tracks_df.to_csv(tracks_file_path, index=False, header=False)


def load_tracks(tracks_file_path: Path) -> List[Track]:

    tracks = []

    tracks_df = pd.read_csv(
        tracks_file_path,
        names=[
            "frame_id",
            "track_id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )

    for track_id, track_data_df in tracks_df.groupby("track_id"):
        trajectory = {}
        for frame_data in track_data_df.itertuples():
            frame_idx = frame_data.frame_id - 1
            position = np.array(
                [
                    frame_data.bb_left,
                    frame_data.bb_top,
                    frame_data.bb_left + frame_data.bb_width,
                    frame_data.bb_top + frame_data.bb_height,
                ]
            )
            trajectory[frame_idx] = position
        track = Track(track_id - 1, class_id=1, trajectory=trajectory)
        tracks.append(track)

    return tracks


def flatten_tracks(tracks: List[Track]):
    mot_tracking_data = []
    for t in tracks:
        for frame_id, bbox in t.trajectory.items():
            mot_tracking_data.append(
                {
                    "frame_id": frame_id,
                    "track_id": t.track_id,
                    "bbox": bbox,
                }
            )

    mot_tracking_data.sort(key=lambda x: (x["frame_id"], x["track_id"]))
    return mot_tracking_data


def index_by(items, key):
    indexed = defaultdict(list)
    for item in items:
        indexed[item[key]].append(item)
    return indexed


def flatten(indexed_items):
    return [item for sublist in indexed_items.values() for item in sublist]
