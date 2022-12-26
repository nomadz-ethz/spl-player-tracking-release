import argparse
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from spltrack.data.annotations import load_csv_annotations


def render_annotated_sequence(
    images_dir_path: Path,
    annotations_file_path: Path,
):
    assert images_dir_path.is_dir()
    assert annotations_file_path.is_file()

    video_file_path = images_dir_path.parent / "annotations_video.avi"

    annotations_per_image = load_csv_annotations(annotations_file_path)

    # Create video writer object
    video_writer = cv.VideoWriter(
        str(video_file_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        15,
        (1920, 1080),
    )
    # Iterate over the images
    for frame_id, image_file_path in enumerate(
        tqdm(sorted(images_dir_path.glob("*.jpg")))
    ):
        if frame_id % 2 == 1:
            continue
        # Load the image
        image = cv.imread(str(image_file_path))
        # Draw all the bounding boxes
        for annotation in annotations_per_image[image_file_path.name]:
            label = annotation["label"]
            color = (255, 0, 0) if label == 1 else (0, 0, 255)
            p1 = (int(annotation["x_min"]), int(annotation["y_min"]))
            p2 = (int(annotation["x_max"]), int(annotation["y_max"]))
            image = cv.rectangle(image, p1, p2, color, 2)
            if label == 0:
                team_number = annotation["color"]
                player_number = annotation["number"]
                cv.putText(
                    image,
                    f"{team_number}-{player_number}",
                    p1,
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        # Add frame to the video
        video_writer.write(image)
    video_writer.release()


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Render annotated sequence of frames to video."
    )

    parser.add_argument(
        "images_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing the frames",
    )

    parser.add_argument(
        "annotations_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the annotations file or CSV format.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    render_annotated_sequence(
        args.images_dir_path,
        args.annotations_file_path,
    )
