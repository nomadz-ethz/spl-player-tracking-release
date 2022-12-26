import xml.etree.ElementTree as ET
from pathlib import Path

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import pandas as pd

from ..utils import ImageSequence, index_by
from .constants import INVERSE_LABEL_MAP, LABEL_MAP


def load_csv_annotations(annotations_file_path: Path):
    annots = pd.read_csv(annotations_file_path).to_dict(orient="records")

    return index_by(annots, "filename")


def save_csv_annotations(annotations_file_path: Path, annotations_per_image):
    annotations = []
    for filename, image_annotations in annotations_per_image.items():
        for a in image_annotations:
            annotations.append({"filename": filename, **a})

    annotations = sorted(
        annotations, key=lambda a: (a["filename"], a["label"], a["color"], a["number"])
    )

    annotations_df = pd.DataFrame.from_records(annotations)
    annotations_df.to_csv(annotations_file_path, header=True, index=False)


def load_cvat_video_annotations(
    cvat_annotations_file_path: Path,
    images_dir_path: Path,
):
    tree = ET.parse(str(cvat_annotations_file_path))
    tracks = tree.findall("track")
    img_seq = ImageSequence(images_dir_path)

    annots = []
    for track_elm in tracks:
        label = INVERSE_LABEL_MAP[track_elm.attrib["label"]]

        for box_elm in track_elm.findall("box"):
            occluded = bool(int(box_elm.attrib["occluded"]))
            outside = bool(int(box_elm.attrib["outside"]))
            if occluded or outside:
                continue
            color = -1
            number = -1
            ignore = 0
            for attr in box_elm.findall("attribute"):
                name = attr.attrib["name"]
                if name == "color":
                    color = int(attr.text)
                elif name == "number":
                    number = int(attr.text)
                elif name == "ignore":
                    ignore = attr.text == "true"
            if label == 0 and (color == -1 or number == -1):
                continue
            frame_id = int(box_elm.attrib["frame"])
            xtl = float(box_elm.attrib["xtl"])
            ytl = float(box_elm.attrib["ytl"])
            xbr = float(box_elm.attrib["xbr"])
            ybr = float(box_elm.attrib["ybr"])
            annots.append(
                {
                    "filename": img_seq[frame_id].name,
                    "label": label,
                    "x_min": xtl,
                    "y_min": ytl,
                    "x_max": xbr,
                    "y_max": ybr,
                    "color": color,
                    "number": number,
                    "ignore": int(ignore),
                }
            )

    annots.sort(key=lambda a: (a["filename"], a["label"], a["color"], a["number"]))

    return index_by(annots, "filename")


def save_cvat_video_annotations(
    annotations_file_path: Path,
    annotations_per_image,
    keyframe_step: int = 1,
):

    num_frames = len(annotations_per_image)

    def populate_metadata_element(meta_elm: ET.Element, name: str):
        task = ET.SubElement(meta_elm, "task")
        task_name = ET.SubElement(task, "name")
        task_name.text = name
        size = ET.SubElement(task, "size")
        size.text = str(num_frames)
        start_frame = ET.SubElement(task, "start_frame")
        start_frame.text = "0"
        stop_frame = ET.SubElement(task, "stop_frame")
        stop_frame.text = str(num_frames - 1)
        source = ET.SubElement(meta_elm, "source")
        source.text = "video.mp4"

    def populate_box_element(box_elm: ET.Element, frame_id, outside, keyframe, bbox):
        box_elm.set("frame", str(frame_id))
        box_elm.set("outside", str(outside))
        box_elm.set("occluded", "0")
        xtl, ytl, xbr, ybr = bbox
        box_elm.set("keyframe", str(keyframe))
        box_elm.set("xtl", str(xtl))
        box_elm.set("ytl", str(ytl))
        box_elm.set("xbr", str(xbr))
        box_elm.set("ybr", str(ybr))
        box_elm.set("z_order", "0")

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
        robot_number_to_track_id[robot_number] = idx
    ball_track_id = len(used_robot_numbers)

    tracking_data = []
    for frame_idx, file_name in enumerate(sorted(list(annotations_per_image.keys()))):
        box_annotations = annotations_per_image[file_name]
        for box_annot in box_annotations:
            jersey_color = box_annot["color"]
            if jersey_color == -1:
                track_id = ball_track_id
                label = "ball"
            else:
                team_offset = team_ids.index(jersey_color) * 5
                robot_number = team_offset + box_annot["number"]
                track_id = robot_number_to_track_id[robot_number]
                label = "robot"

            tracking_data.append(
                {
                    "frame_id": frame_idx,
                    "track_id": track_id,
                    "label": label,
                    "bbox": (
                        box_annot["x_min"],
                        box_annot["y_min"],
                        box_annot["x_max"],
                        box_annot["y_max"],
                    ),
                }
            )

    tracking_data_by_track_id = index_by(tracking_data, "track_id")
    root = ET.Element("annotations")
    meta = ET.SubElement(tree, "meta")
    populate_metadata_element(meta, annotations_file_path.stem)
    for track_id, tds in tracking_data_by_track_id.items():
        if len(tds) < 5:
            continue
        tds.sort(key=lambda x: x["frame_id"])
        track_element = ET.SubElement(tree, "track")
        track_element.set("id", str(track_id))
        track_element.set("label", "robot")
        track_element.set("source", "manual")
        for td in tds:
            box_element = ET.SubElement(track_element, "box")
            frame_id = td["frame_id"]
            bbox = td["bbox"]
            keyframe = 1 if frame_id % keyframe_step == 0 else 0
            populate_box_element(
                box_element,
                frame_id=frame_id,
                outside=0,
                keyframe=keyframe,
                bbox=bbox,
            )

        last_td = max(tds, key=lambda x: x["frame_id"])
        last_frame_id = last_td["frame_id"]
        if last_frame_id < 4999:
            box_element = ET.SubElement(track_element, "box")
            last_bbox = last_td["bbox"]
            populate_box_element(
                box_element,
                frame_id=frame_id + 1,
                outside=1,
                keyframe=1,
                bbox=last_bbox,
            )

    tree = ET.ElementTree(root)

    xmlstr = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(indent="    ")
    with open(annotations_file_path, "w") as f:
        f.write(xmlstr)
