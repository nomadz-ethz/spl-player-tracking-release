import json
from pathlib import Path

import numpy as np

from trackeval.datasets._base_dataset import _BaseDataset
from trackeval import utils

from ..data.constants import (
    ORC_SPLITS,
    INVERSE_LABEL_MAP,
    TEAM_NUMBER_TO_COLORS,
)
from ..data.annotations import load_csv_annotations
from ..data.utils import bbox_from_points
from ..game_data import GameState
from ..utils import index_by, load_tracks, flatten_tracks


_BALL_TRACK_ID = 11


class ORCTrackEvalDataset(_BaseDataset):
    @staticmethod
    def get_default_dataset_config():
        default_config = {
            "GT_FOLDER": "datasets/orc_rc22/",  # Location of GT data
            "TRACKERS_FOLDER": "trackers/orc_rc22/",  # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": ["robot"],
            "SPLIT_TO_EVAL": "val",  # Valid: 'train', 'val, 'test', 'all'
            "PRINT_CONFIG": True,  # Whether to print current config
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            "MATCH_PLAYERS": False,  # Match the track id of the tracklets i.e. treat it as a class label
            # "SEQMAP_FILE": None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            # "SEQ_INFO": None,  # If not None, directly specify sequences to eval and their number of timesteps
            # "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",  # '{gt_folder}/{seq}/gt/gt.txt'
            # "SKIP_SPLIT_FOL": False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        if self.config["SPLIT_TO_EVAL"] == "all":
            self.seq_list = ORC_SPLITS["train"] + ORC_SPLITS["val"] + ORC_SPLITS["test"]
        else:
            self.seq_list = ORC_SPLITS[self.config["SPLIT_TO_EVAL"]]

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = [
                p.name for p in Path(self.tracker_fol).iterdir() if p.is_dir()
            ]
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        self.match_players = self.config["MATCH_PLAYERS"]
        if self.match_players:
            self.class_list = [f"home_{i+1}" for i in range(5)] + [
                f"away_{i+1}" for i in range(5)
            ]
        else:
            self.class_list = self.config["CLASSES_TO_EVAL"]

    def _load_raw_file(self, tracker, seq, is_gt):

        # FIXME: this will throw an exception if not found, should be read from elsewhere
        team_info = GameState.read_from_json(
            next(Path(self.gt_fol).joinpath(seq).joinpath("gc/json").glob("*.g.json"))
        ).get_team_info()
        color_to_team_idx = {
            TEAM_NUMBER_TO_COLORS[team_info["home"]["team_number"]][0]: 0,
            TEAM_NUMBER_TO_COLORS[team_info["away"]["team_number"]][0]: 1,
        }

        if is_gt:
            raw_data = {
                "gt_dets": [],
                "gt_ids": [],
                "gt_classes": [],
            }
            annotations_file_path = next(Path(self.gt_fol).joinpath(seq).glob("*.csv"))
            annotations_per_image = load_csv_annotations(annotations_file_path)

            for image_file_name in sorted(annotations_per_image.keys()):
                annots = annotations_per_image[image_file_name]
                dets = []
                ids = []
                classes = []
                for a in annots:
                    if a["label"] == INVERSE_LABEL_MAP["ball"]:
                        # FIXME: ignore balls for now
                        # classes.append(self.class_list.index("ball"))
                        # ids.append(_BALL_TRACK_ID)
                        continue
                    else:
                        color = a["color"]
                        number = a["number"]
                        track_id = color_to_team_idx[color] * 5 + number - 1
                        ids.append(track_id)
                        if self.match_players:
                            classes.append(track_id)
                        else:
                            classes.append(self.class_list.index("robot"))

                    x1, y1, x2, y2 = a["x_min"], a["y_min"], a["x_max"], a["y_max"]
                    dets.append(bbox_from_points(x1, y1, x2, y2, mode="xywh"))

                raw_data["gt_dets"].append(np.array(dets))
                raw_data["gt_ids"].append(np.array(ids))
                raw_data["gt_classes"].append(np.array(classes))
            raw_data["num_timesteps"] = len(annotations_per_image)
        else:
            raw_data = {
                "tracker_dets": [],
                "tracker_ids": [],
                "tracker_classes": [],
                "tracker_confidences": [],
            }
            tracks_file_path = Path(self.tracker_fol) / tracker / seq / "tracks.csv"
            tracking_data = flatten_tracks(load_tracks(tracks_file_path))
            tracking_data_by_frame = index_by(tracking_data, "frame_id")
            for frame_id in sorted(tracking_data_by_frame.keys()):
                tds = tracking_data_by_frame[frame_id]
                dets = []
                ids = []
                classes = []
                for td in tds:
                    x1, y1, x2, y2 = td["bbox"]
                    dets.append(bbox_from_points(x1, y1, x2, y2, mode="xywh"))
                    ids.append(td["track_id"])
                    if self.match_players:
                        classes.append(td["track_id"])
                    else:
                        classes.append(td.get("label", 0))
                raw_data["tracker_dets"].append(np.array(dets))
                raw_data["tracker_ids"].append(np.array(ids))
                raw_data["tracker_classes"].append(np.array(classes))
                raw_data["tracker_confidences"].append(
                    np.ones((len(classes),), dtype=np.float32)
                )
            # FIXME: should read this from sequence metadata!
            raw_data["num_timesteps"] = 5000
        raw_data["seq"] = seq
        return raw_data

    def get_preprocessed_seq_data(self, raw_data, cls):

        class_id = self.class_list.index(cls)

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}

        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):

            # Get all data
            gt_ids = raw_data["gt_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            gt_classes = raw_data["gt_classes"][t]

            similarity_scores = raw_data["similarity_scores"][t]

            data["tracker_ids"][t] = raw_data["tracker_ids"][t]
            data["tracker_dets"][t] = raw_data["tracker_dets"][t]
            data["tracker_confidences"][t] = raw_data["tracker_confidences"][t]

            gt_to_keep_mask = np.equal(gt_classes, class_id)
            data["gt_ids"][t] = gt_ids[gt_to_keep_mask]
            data["gt_dets"][t] = gt_dets[gt_to_keep_mask, :]
            data["similarity_scores"][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(np.int)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(np.int)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="xywh"
        )
        return similarity_scores
