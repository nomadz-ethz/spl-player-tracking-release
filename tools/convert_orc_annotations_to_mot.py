import argparse
import csv
from pathlib import Path

from spltrack.data.annotations import load_csv_annotations
from spltrack.data.constants import LABEL_MAP, TRACKING_SPLITS
from spltrack.data.utils import find_sequence_dirs


def main(orc_dataset_dir_path: Path, include_ball: bool = False):

    sequences = find_sequence_dirs(orc_dataset_dir_path)

    for sequence_dir_path in sequences:
        annotations_file_path = next(sequence_dir_path.glob("*.csv"))
        annotations_per_image = load_csv_annotations(annotations_file_path)

        gt_dir_path = sequence_dir_path / "gt"
        gt_dir_path.mkdir(exist_ok=True)

        with open(sequence_dir_path.joinpath("seqinfo.ini"), "w") as f:
            f.writelines(
                [
                    s + "\n"
                    for s in [
                        "[Sequence]",
                        f"name={sequence_dir_path.name}",
                        "imDir=images",
                        "frameRate=30",
                        "seqLength=5000",
                        "imWidth=1920",
                        "imHeight=1080",
                        "imExt=.jpg",
                    ]
                ]
            )

        # Get team ids
        team_ids = set()
        for box_annot in annotations_per_image[list(annotations_per_image.keys())[0]]:
            if LABEL_MAP[box_annot["label"]] == "robot":
                team_ids.add(box_annot["color"])
                if len(team_ids) > 1:
                    break
        team_ids = sorted(list(team_ids))

        used_robot_numbers = set()
        for _, box_annotations in annotations_per_image.items():
            for box_annot in box_annotations:
                jersey_color = box_annot["color"]
                if box_annot["color"] == -1:
                    continue
                team_offset = team_ids.index(jersey_color) * 5
                used_robot_numbers.add(team_offset + box_annot["number"])

            if len(used_robot_numbers) == 10:
                break

        robot_number_to_track_id = {}
        for idx, robot_number in enumerate(sorted(list(used_robot_numbers))):
            robot_number_to_track_id[robot_number] = idx + 1
        ball_track_id = len(used_robot_numbers) + 1

        with open(gt_dir_path.joinpath("gt.txt"), "w") as f:
            csv_writer = csv.DictWriter(
                f,
                fieldnames=[
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
            for frame_idx, file_name in enumerate(
                sorted(list(annotations_per_image.keys()))
            ):
                mot_annotations = []
                box_annotations = annotations_per_image[file_name]
                for box_annot in box_annotations:
                    jersey_color = box_annot["color"]
                    if jersey_color == -1:
                        if include_ball:
                            track_id = ball_track_id
                        else:
                            continue
                    else:
                        team_offset = team_ids.index(jersey_color) * 5
                        robot_number = team_offset + box_annot["number"]
                        track_id = robot_number_to_track_id[robot_number]
                    mot_annotations.append(
                        {
                            "frame_id": frame_idx + 1,
                            "track_id": track_id,
                            "bb_left": box_annot["x_min"],
                            "bb_top": box_annot["y_min"],
                            "bb_width": box_annot["x_max"] - box_annot["x_min"],
                            "bb_height": box_annot["y_max"] - box_annot["y_min"],
                            "conf": -1,
                            "x": -1,
                            "y": -1,
                            "z": -1,
                        }
                    )
                mot_annotations = sorted(mot_annotations, key=lambda x: x["track_id"])
                for mot_annot in mot_annotations:
                    csv_writer.writerow(mot_annot)

    seqmaps_dir_path = orc_dataset_dir_path / "seqmaps"
    seqmaps_dir_path.mkdir(exist_ok=True)
    for split in ["train", "val"]:
        seqmap_file_path = seqmaps_dir_path / f"{orc_dataset_dir_path.name}-{split}.txt"
        with seqmap_file_path.open("w") as f:
            f.write("name\n")
            f.writelines([s + "\n" for s in TRACKING_SPLITS[split]])


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "orc_dataset_dir",
        type=lambda p: Path(p).resolve(strict=True),
        help="Path to the orc dataset dir",
    )

    parser.add_argument(
        "--include_ball",
        action="store_true",
        help="If the ball track should be included.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.orc_dataset_dir, args.include_ball)
