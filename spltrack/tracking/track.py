from typing import Dict, Optional

import numpy as np

from ..data.constants import INVERSE_LABEL_MAP


class Track:
    def __init__(
        self,
        track_id: int,
        class_id: int,
        position: Optional[np.ndarray] = None,
        start_frame_id: Optional[int] = None,
        trajectory: Optional[Dict[int, np.ndarray]] = None,
    ):

        self.track_id = track_id
        self.class_id = class_id
        if position is not None and start_frame_id is not None:
            self.start_frame_id = start_frame_id
            self.last_frame_id = start_frame_id
            self.current_pos = position
            self.trajectory = {start_frame_id: position}
        elif trajectory is not None:
            start_frame_id, last_frame_id = min(trajectory.keys()), max(
                trajectory.keys()
            )
            if start_frame_id > last_frame_id:
                raise ValueError("Invalid trajectory")
            for frame_id, pos in trajectory.items():
                if pos[0] >= pos[2] or pos[1] >= pos[3]:
                    raise ValueError(
                        f"Trajectory has invalid pose at frame {frame_id}!"
                    )
            self.trajectory = trajectory
            self.start_frame_id = start_frame_id
            self.last_frame_id = last_frame_id
            self.current_pos = self.trajectory[last_frame_id]
        else:
            raise ValueError(
                "Either position and start frame id or trajectory "
                "must be provided to initialize a track."
            )
        self.inactive_count = 0

    @property
    def current_pos(self) -> np.ndarray:
        return self._current_pos

    @current_pos.setter
    def current_pos(self, pos: np.ndarray):
        if pos[0] >= pos[2] or pos[1] >= pos[3]:
            raise ValueError("Invalid position!")
        self._current_pos = pos

    @property
    def class_id(self) -> int:
        return self._class_id

    @class_id.setter
    def class_id(self, class_id: int):
        if hasattr(self, "_class_id"):
            raise ValueError("The class id of a track cannot be modified!")
        self._class_id = class_id

    def update(self, frame_id, new_pos: Optional[np.ndarray] = None):
        if new_pos is None:
            self.inactive_count += 1
        else:
            self.inactive_count = 0
            self.current_pos = new_pos
        self.last_frame_id = frame_id
        self.trajectory[frame_id] = self.current_pos.copy()


class PlayerTrack(Track):
    def __init__(
        self,
        track_id: int,
        position: Optional[np.ndarray] = None,
        start_frame_id: Optional[int] = None,
        trajectory: Optional[Dict[int, np.ndarray]] = None,
    ):
        super().__init__(
            track_id=track_id,
            class_id=INVERSE_LABEL_MAP["robot"],
            position=position,
            start_frame_id=start_frame_id,
            trajectory=trajectory,
        )
        self.number_detections = dict()
        self.team_detections = dict()

    @classmethod
    def from_track(cls, track: Track):
        if track.class_id != INVERSE_LABEL_MAP["robot"]:
            raise ValueError("Cannot create PlayerTrack from Track not of class robot!")
        return cls(
            track_id=track.track_id,
            trajectory=track.trajectory,
        )

    def add_number_detection(self, frame_id: int, number: int):
        self.number_detections[frame_id] = number

    def add_team_detection(self, frame_id, team: int):
        self.team_detections[frame_id] = team
