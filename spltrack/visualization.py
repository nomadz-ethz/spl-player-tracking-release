import bisect
from copy import deepcopy
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, Iterable, List, Tuple, Union
import cv2 as cv
import functools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tqdm import tqdm

from .data.constants import COLOR_TO_RGB, LABEL_MAP
from .utils import flatten_tracks, index_by


@dataclass
class Fragment:
    # A piece of text with its own style
    text: Any = None
    fontFace: int = None
    fontScale: float = None
    color: Tuple[int] = None
    thickness: int = None
    lineType: int = None
    bottomLeftOrigin: bool = None

    @staticmethod
    def default():
        return Fragment(
            text="",
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv.LINE_8,
            bottomLeftOrigin=False,
        )

    def combine(self, other: "Fragment") -> "Fragment":
        new_fragment = deepcopy(self)
        for k, v in dataclasses.asdict(other).items():
            if v is not None:
                setattr(new_fragment, k, v)
        return new_fragment

    def write(self, image, org):
        if str(self.text) != "":
            cv.putText(
                image,
                str(self.text),
                org,
                self.fontFace,
                self.fontScale,
                self.color,
                self.thickness,
                self.lineType,
                self.bottomLeftOrigin,
            )

    def text_size(self):
        if not self.text:
            return (0, 0), 0
        return cv.getTextSize(
            str(self.text), self.fontFace, self.fontScale, self.thickness
        )

    def box_size(self):
        (w, h), baseline = self.text_size()
        return w, h + baseline


def write_text(
    image: np.ndarray,
    org,
    fragments: Iterable[Union[str, Fragment]] = [],
    *,
    offset: Tuple[int, int] = (0, 0),
    style: Fragment = Fragment(),
) -> tuple:
    # org: origin (x, y)
    # offset: additive offset to org
    # fragments: list of strings (text) or Fragments; overrides "style"
    # style: default styles to apply
    # returns size (w, h)
    (x, y), (ofx, ofy) = org, offset
    x, y = x + ofx, y + ofy
    w, h = 0, 0
    style = Fragment.default().combine(style)
    try:
        iter(fragments)
    except TypeError:
        fragments = [fragments]
    for fragment in fragments:
        if isinstance(fragment, Fragment):
            fragment = style.combine(fragment)
        else:
            fragment = style.combine(Fragment(text=fragment))
        fragment.write(image, (x, y))
        (tw, th), baseline = fragment.text_size()
        x, y = x + tw, y
        w, h = w + tw, max(h, th + baseline)
    return w, h


def write_table(
    image,
    org,
    table: List[List[Union[Fragment, Any]]],
    style: Fragment = Fragment(),
    *,
    cell_spacing: Union[int, Tuple[int, int]] = 0,
):
    if isinstance(cell_spacing, int):
        cell_spacing = (cell_spacing, cell_spacing)
    spacing_w, spacing_h = cell_spacing
    fragments_table = [
        [
            style.combine(cell if isinstance(cell, Fragment) else Fragment(text=cell))
            for cell in row
        ]
        for row in table
    ]
    # sizes: (rows, cols, 2)
    sizes = np.array(
        [[fragment.box_size() for fragment in row] for row in fragments_table]
    )
    sizes += np.array([spacing_w * 2, spacing_h * 2])[None, None]
    col_ws, row_hs = sizes[:, :, 0].max(axis=0), sizes[:, :, 1].max(axis=1)
    col_xs, row_ys = np.cumsum([0] + list(col_ws)), np.cumsum([0] + list(row_hs))
    x, y = org
    for i, row in enumerate(fragments_table):
        for j, fragment in enumerate(row):
            fragment.write(
                image, (x + col_xs[j] + spacing_w, y + row_ys[i] + spacing_h)
            )
    return col_xs[-1], row_ys[-1]


def draw_tracks(image, track_data, color):
    for t in track_data:
        x1, y1, x2, y2 = t["bbox"].astype(int)
        track_id = t["track_id"]
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            image,
            f"{track_id}",
            (int(x1), int(y1)),
            cv.FONT_HERSHEY_PLAIN,
            1.8,
            color,
            2,
        )


def draw_team_probs(image, probs, x, y, w, h):
    xs = [0] + list(np.cumsum(w * probs).round().astype(int))
    for team_color, (cx1, cx2) in enumerate(zip(xs[:-1], xs[1:])):
        cv.rectangle(
            image, (x + cx1, y), (x + cx2, y + h), COLOR_TO_RGB[team_color], cv.FILLED
        )


class TrackRenderer:
    def __init__(
        self,
        framerate,
        image_resolution,
        footer_height=0,
        framerate_divider=2,
        downscale_factor=2,
        max_workers=4,
    ):
        assert isinstance(framerate_divider, int) and framerate_divider > 0
        self.framerate_divider = framerate_divider
        assert isinstance(downscale_factor, int) and downscale_factor > 0
        self.downscale_factor = downscale_factor
        self.framerate = framerate // framerate_divider
        self.output_resolution = (
            image_resolution[0] // downscale_factor,
            (image_resolution[1] + footer_height) // downscale_factor,
        )
        self.footer_height = footer_height // downscale_factor
        self.max_workers = max_workers

    @staticmethod
    def _render_frame(frame_data, output_resolution, footer_height, team_info):
        (
            image_file_path,
            track_data,
            excluded_track_data,
            tracklet_data,
            track_id_to_player,
            team_comm_data,
            tracking_data,
            tracklet_to_track_cost_details,
            tracklet_to_track_id,
            tracklet_to_tracklet_costs,
            annotation_data,
        ) = frame_data

        image = cv.imread(str(image_file_path))
        h, w, c = image.shape
        unresized_footer_height = round(
            h / (output_resolution[1] - footer_height) * footer_height
        )
        image = np.concatenate(
            [image, np.full((unresized_footer_height, w, c), 255, dtype=np.uint8)]
        )
        footer_y = h

        # set all excluded track ids to -1
        for t in excluded_track_data:
            x1, y1, x2, y2 = t["bbox"].astype(int)
            color = COLOR_TO_RGB[-1]
            cv.putText(image, "?", (x1, y1 - 6), cv.FONT_HERSHEY_PLAIN, 1.8, color, 2)
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            for tdata in tracking_data:
                if np.allclose(tdata["bbox"], t["bbox"]):
                    draw_team_probs(image, tdata["team_probs"], x1, y1 - 6, 50, 3)
                    break

        # Get tracks for this frame
        for t in track_data:
            team, player_number = track_id_to_player[t["track_id"]]
            team_color = (
                team_info["home"]["team_color"]
                if team_info["home"]["team_number"] == team
                else team_info["away"]["team_color"]
            )
            team_rgb = COLOR_TO_RGB[team_color]
            x1, y1, x2, y2 = t["bbox"].astype(int)
            px, py = (x1 + x2) // 2, y2
            for ann in annotation_data:
                if (ann["color"], ann["number"]) == (team_color, player_number):
                    ax, ay = (int(ann["x_min"]) + int(ann["x_max"])) // 2, int(
                        ann["y_max"]
                    )
                    cv.line(image, (ax, ay), (px, py), team_rgb, 2)
                    break
            cv.putText(
                image,
                f"{team_color}:{player_number}",
                (x1, y1 - 6),
                cv.FONT_HERSHEY_PLAIN,
                1.8,
                team_rgb,
                2,
            )
            cv.rectangle(image, (x1, y1), (x2, y2), team_rgb, 2)
            for tdata in tracking_data:
                if np.allclose(tdata["bbox"], t["bbox"]):
                    draw_team_probs(image, tdata["team_probs"], x1, y1 - 6, 40, 3)
                    break

        for t in tracklet_data:
            x1, y1, x2, y2 = t["bbox"].astype(int)
            text = str(t["track_id"])
            if t["fallen_player"] is not None:
                ftn, fpn = t["fallen_player"]
                text += f"fallen: {ftn}:{fpn}"
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            cv.putText(
                image,
                text,
                (int(x1), int(y1) + 20),
                cv.FONT_HERSHEY_PLAIN,
                1.8,
                color,
                2,
            )

        for pose in team_comm_data.values():
            team, player_number = pose["player"]
            team_color = (
                team_info["home"]["team_color"]
                if team_info["home"]["team_number"] == team
                else team_info["away"]["team_color"]
            )
            px, py = int(pose["ix"]), int(pose["iy"])
            for ann in annotation_data:
                if (ann["color"], ann["number"]) == (team_color, player_number):
                    ax, ay = (int(ann["x_min"]) + int(ann["x_max"])) // 2, int(
                        ann["y_max"]
                    )
                    cv.line(image, (ax, ay), (px, py), (255, 255, 255))
                    break
            cv.drawMarker(image, (px, py), (255, 255, 255), cv.MARKER_CROSS)
            cv.putText(
                image,
                f"{team_color}:{player_number}",
                (px, py),
                cv.FONT_HERSHEY_PLAIN,
                1.8,
                (255, 255, 255),
                2,
            )

        # Set up table headers for columns (tracklets) & rows (track IDs)
        visible_tracklet_ids = sorted(
            {tracklet["track_id"] for tracklet in tracklet_data}
        )
        track_ids = sorted(
            {track_id for (_, track_id) in tracklet_to_track_cost_details.keys()}
        )
        cols = visible_tracklet_ids
        rows = track_ids
        table = [[""] * (len(cols) + 1) for _ in range(len(rows) + 1)]
        x, y = (0, footer_y)
        w, h = write_text(image, (x, y), "\n")
        x, y = x, y + h
        for i, track_id in enumerate(rows):
            # Row headers
            team, player_number = track_id_to_player[track_id]
            team_color = (
                team_info["home"]["team_color"]
                if team_info["home"]["team_number"] == team
                else team_info["away"]["team_color"]
            )
            team_rgb = COLOR_TO_RGB[team_color]
            table[i + 1][0] = Fragment(
                f"{team_color}:{player_number}",
                color=team_rgb,
                thickness=(
                    2
                    if any(
                        tr == track_id
                        for trl, tr in tracklet_to_track_id.items()
                        if trl in visible_tracklet_ids
                    )
                    else 1
                ),
            )
        for j, tracklet_id in enumerate(cols):
            # Column headers
            table[0][j + 1] = Fragment(
                tracklet_id,
                color=(255, 255, 0),
                thickness=(
                    2
                    if tracklet_id in tracklet_to_track_id.keys()
                    and tracklet_to_track_id[tracklet_id] != -1
                    else 1
                ),
            )

        # Localization costs only
        for (tracklet_id, track_id), costs in tracklet_to_track_cost_details.items():
            if tracklet_id not in cols:
                continue
            i, j = rows.index(track_id) + 1, cols.index(tracklet_id) + 1
            value = round(costs["loc"] * 1000)
            if tracklet_to_track_id[tracklet_id] == track_id:
                table[i][j] = Fragment(value, color=(0, 0, 0))
            else:
                table[i][j] = round(value, 2)
        tw, th = write_table(
            image,
            (x, y),
            table,
            style=Fragment(
                fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(99, 99, 99)
            ),
            cell_spacing=(3, 0),
        )
        write_text(image, (x, y + th + 5), "cost[loc] * 1000")

        # Team detection costs only
        x, y = x + tw + 20, y
        for (tracklet_id, track_id), costs in tracklet_to_track_cost_details.items():
            if tracklet_id not in cols:
                continue
            i, j = rows.index(track_id) + 1, cols.index(tracklet_id) + 1
            value = round(costs["team"] * 1000)
            if tracklet_to_track_id[tracklet_id] == track_id:
                table[i][j] = Fragment(value, color=(0, 0, 0))
            else:
                table[i][j] = value
        tw, th = write_table(
            image,
            (x, y),
            table,
            style=Fragment(
                fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(99, 99, 99)
            ),
            cell_spacing=(3, 0),
        )
        write_text(image, (x, y + th + 5), "cost[team] * 1000")

        # Track length costs only
        x, y = x + tw + 20, y
        for (tracklet_id, track_id), costs in tracklet_to_track_cost_details.items():
            if tracklet_id not in cols:
                continue
            i, j = rows.index(track_id) + 1, cols.index(tracklet_id) + 1
            value = round(costs["len"] * 1000)
            if tracklet_to_track_id[tracklet_id] == track_id:
                table[i][j] = Fragment(value, color=(0, 0, 0))
            else:
                table[i][j] = value
        tw, th = write_table(
            image,
            (x, y),
            table,
            style=Fragment(
                fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(99, 99, 99)
            ),
            cell_spacing=(3, 0),
        )
        write_text(image, (x, y + th + 5), "cost[len] * 1000")

        x, y = 0, y + th + 40
        # rows: currently visible tracklets
        # cols: any other tracklets paired with currently visible tracklets
        rows = set()
        paired_tracklet_ids = set()
        for tracklet1, tracklet2, _ in tracklet_to_tracklet_costs:
            if tracklet2 in visible_tracklet_ids:
                rows.add(tracklet2)
                paired_tracklet_ids.add(tracklet1)
            if tracklet1 in visible_tracklet_ids:
                rows.add(tracklet1)
                paired_tracklet_ids.add(tracklet2)
        paired_tracklet_ids = sorted(paired_tracklet_ids)
        rows = sorted(rows)
        cols = paired_tracklet_ids
        table = [[""] * (len(cols) + 1) for _ in range(len(rows) + 1)]
        for i, tracklet_id in enumerate(rows):
            table[i + 1][0] = Fragment(
                tracklet_id,
                color=(255, 255, 0),
                thickness=(
                    2
                    if tracklet_id in tracklet_to_track_id.keys()
                    and tracklet_to_track_id[tracklet_id] != -1
                    else 1
                ),
            )
        for j, tracklet_id in enumerate(cols):
            table[0][j + 1] = Fragment(
                tracklet_id,
                color=(255, 255, 0),
                thickness=(
                    2
                    if tracklet_id in tracklet_to_track_id.keys()
                    and tracklet_to_track_id[tracklet_id] != -1
                    else 1
                ),
            )
        for tracklet1, tracklet2, cost in tracklet_to_tracklet_costs:
            # make tracklet1 the row (i.e. visible tracklet), tracklet2 the column
            if (
                tracklet1 not in visible_tracklet_ids
                and tracklet2 not in visible_tracklet_ids
            ):
                continue
            if tracklet2 in visible_tracklet_ids:
                tracklet1, tracklet2 = tracklet2, tracklet1
            i, j = rows.index(tracklet1) + 1, cols.index(tracklet2) + 1
            value = round(cost * 1000)
            if (
                tracklet_to_track_id[tracklet1] == tracklet_to_track_id[tracklet2]
                and tracklet_to_track_id[tracklet1] != -1
            ):
                table[i][j] = Fragment(value, color=(0, 0, 0))
            else:
                table[i][j] = value
        extracols = [""]
        for tracklet_id in cols:
            track_id = tracklet_to_track_id[tracklet_id]
            if track_id == -1:
                extracols.append("")
            else:
                team, player_number = track_id_to_player[track_id]
                team_color = (
                    team_info["home"]["team_color"]
                    if team_info["home"]["team_number"] == team
                    else team_info["away"]["team_color"]
                )
                team_rgb = COLOR_TO_RGB[team_color]
                extracols.append(
                    Fragment(
                        f"{team_color}:{player_number}",
                        color=team_rgb,
                        thickness=1,
                    )
                )
        table = [extracols] + table
        tw, th = write_table(
            image,
            (x, y),
            table,
            style=Fragment(
                fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(99, 99, 99)
            ),
            cell_spacing=(3, 0),
        )
        write_text(image, (x, y + th + 5), "pairwise costs * 1000")

        for ann in annotation_data:
            x1, y1, x2, y2 = ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            px, py = (x1 + x2) // 2, y2
            rgb = (0, 255, 0)
            cv.line(image, (x1, y2), (x2, y2), rgb, 1)
            cv.drawMarker(image, (px, py), rgb, cv.MARKER_CROSS)
            cv.putText(
                image,
                f"{ann['color']}:{ann['number']}",
                (px, py),
                cv.FONT_HERSHEY_PLAIN,
                1.5,
                rgb,
                2,
            )

        return cv.resize(image, output_resolution)

    def render(
        self,
        video_file_path,
        image_sequence,
        tracks,
        excluded_tracks=[],
        tracklets=[],
        track_id_to_player={},
        team_info={},
        team_comm_poses_by_ts={},
        tracking_data=[],
        tracklet_to_fallen_events={},
        tracklet_to_track_cost_details={},
        tracklet_to_track_id={},
        tracklet_to_tracklet_costs=[],
        annotations={},
    ):
        track_data = flatten_tracks(tracks)
        track_data_by_frame = index_by(track_data, "frame_id")

        excluded_track_data = flatten_tracks(excluded_tracks)
        excluded_track_data_by_frame = index_by(excluded_track_data, "frame_id")

        tracklet_data = flatten_tracks(tracklets)
        tracklet_data_by_frame = index_by(tracklet_data, "frame_id")

        frame_ts = [
            image_sequence.get_frame_timestamp(frame_id)
            for frame_id in range(len(image_sequence))
        ]
        assert frame_ts == sorted(frame_ts)

        team_comm_poses_by_player_by_frame = defaultdict(dict)
        players = set()
        for ts, poses in team_comm_poses_by_ts.items():
            if ts < frame_ts[0]:
                continue
            frame_id = bisect.bisect_left(frame_ts, ts)

            for pose in poses:
                player = pose["player"]
                players.add(player)
                team_comm_poses_by_player_by_frame[frame_id][player] = pose

        # Going forward in time, fill gaps in team comm pose data for each player
        last_pose_by_player = {}
        for player in players:
            for frame_id in range(len(image_sequence)):
                frame_data = team_comm_poses_by_player_by_frame[frame_id]
                if player in frame_data:
                    last_pose_by_player[player] = frame_data[player]
                elif player in last_pose_by_player:
                    frame_data[player] = last_pose_by_player[player]
                else:
                    continue

        tracking_data_per_frame = index_by(tracking_data, "frame_id")

        annotations_per_frame = {
            image_sequence.get_frame_id(int(Path(filename).stem)): [
                annotation
                for annotation in image_annotations
                if LABEL_MAP[annotation["label"]] == "robot"
            ]
            for filename, image_annotations in annotations.items()
        }
        assert None not in annotations_per_frame

        # Add fallen info to tracklets data
        for frame_id, tds in tracklet_data_by_frame.items():
            for td in tds:
                td["fallen_player"] = None
                if td["track_id"] in tracklet_to_fallen_events:
                    for fe in tracklet_to_fallen_events[td["track_id"]]:
                        if (
                            frame_ts[frame_id] >= fe["start_timestamp"]
                            and frame_ts[frame_id] <= fe["end_timestamp"]
                        ):
                            td["fallen_player"] = (
                                fe["team_number"],
                                fe["player_number"],
                            )
                            break

        data = []
        for frame_id, image_file_path in enumerate(image_sequence):
            if frame_id % self.framerate_divider != 0:
                continue
            ts = image_sequence.get_frame_timestamp(frame_id)
            data.append(
                (
                    image_file_path,
                    track_data_by_frame.get(frame_id, []),
                    excluded_track_data_by_frame.get(frame_id, []),
                    tracklet_data_by_frame.get(frame_id, []),
                    track_id_to_player,
                    team_comm_poses_by_player_by_frame.get(frame_id, {}),
                    tracking_data_per_frame.get(frame_id, []),
                    tracklet_to_track_cost_details,
                    tracklet_to_track_id,
                    tracklet_to_tracklet_costs,
                    annotations_per_frame.get(frame_id, []),
                )
            )

        video_writer = cv.VideoWriter(
            str(video_file_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            self.framerate,
            self.output_resolution,
        )
        render_worker = functools.partial(
            TrackRenderer._render_frame,
            output_resolution=self.output_resolution,
            footer_height=self.footer_height,
            team_info=team_info,
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for image in list(tqdm(executor.map(render_worker, data), total=len(data))):
                video_writer.write(image)
        video_writer.release()
