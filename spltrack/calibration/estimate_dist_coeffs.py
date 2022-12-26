import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from ..utils import ImageSequence
from .utils import compute_median_image


_ASCII_TABLE = {i: chr(i) for i in range(128)}


def _save_line_segment_points(line_segments, output_file_path):
    with open(output_file_path, "w") as f:
        f.write(f"{len(line_segments)}\n")
        for line_segment_points in line_segments:
            f.write(f"{len(line_segment_points)}\n")
            for point in line_segment_points:
                f.write(f"{point[0]:.5f} {point[1]:.5f}\n")


def _pick_line_segment_points(image):
    orig_image = np.copy(image)
    cv.putText(
        orig_image,
        "Click line segment points. Press 'n' to finish a line. Press 'c' to clear. Press 'q' to finish.",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    result_image = np.copy(orig_image)
    window_name = "Calibration image"
    lines = []
    current_line_points = []
    points = []

    def click_event_cb(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(result_image, (x, y), 5, (0, 0, 255), 2)
            current_line_points.append((x, y))
            cv.imshow(window_name, result_image)

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, click_event_cb)

    try:
        k = 0
        # Close the window when key q is pressed
        while _ASCII_TABLE.get(k, None) != "q":
            # Display the image
            cv.imshow(window_name, result_image)
            k = cv.waitKey(0)
            # If c is pressed, clear the window, using the dummy image
            if _ASCII_TABLE.get(k, None) == "c":
                result_image = np.copy(orig_image)
                points.clear()
                current_line_points.clear()
                cv.imshow(window_name, result_image)
            elif _ASCII_TABLE.get(k, None) == "n":
                if len(current_line_points) >= 2:
                    start = current_line_points[0]
                    end = current_line_points[-1]
                    cv.line(result_image, start, end, (255, 0, 0), 2)
                    cv.imshow(window_name, result_image)
                    lines.append(current_line_points.copy())
                current_line_points.clear()
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()

    return lines


def parse_dist_coeffs(lens_distortion_estimation_output_file_path: Path) -> np.ndarray:
    with open(lens_distortion_estimation_output_file_path, "r") as f:
        lines = f.readlines()

    index = next(
        index
        for index, line in reversed(list(enumerate(lines)))
        if "Distortion parameters" in line
    )

    dist_coeffs = []
    for line in lines[index + 1 : index + 6]:
        dist_coeffs.append(float(line.split("=")[1].lstrip(" ").rstrip("\n")))

    return np.array(dist_coeffs)


def main(images_dir_path: Path, output_dir_path: Optional[Path]):

    # first check that the lens distortion executable exists
    lens_distortion_estimation_executable_path = (
        Path(__file__).parents[2] / "bin/lens_distortion_estimation"
    )
    if not lens_distortion_estimation_executable_path.is_file():
        print("Lens distortion estimation executable not found!")
        return

    print("Loading image sequence")
    image_sequence = ImageSequence(images_dir_path=images_dir_path)

    if output_dir_path is None:
        output_dir_path = images_dir_path.parent

    median_image_file_path = output_dir_path / "median.bmp"
    if median_image_file_path.is_file():
        print("Loading existing median image.")
        median_image = cv.imread(str(median_image_file_path))
    else:
        print("Computing median image.")
        median_image = compute_median_image(image_sequence)
        cv.imwrite(str(median_image_file_path), median_image)

    # create temporary directory for storing intermediate results
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        line_segment_points = _pick_line_segment_points(median_image)
        line_segment_points_file_path = tmp_dir_path / "distorted_line_segments.dat"
        _save_line_segment_points(line_segment_points, line_segment_points_file_path)

        undistorted_image_file_path = output_dir_path / "median.undistorted.bmp"
        lens_distortion_estimation_output_file_path = (
            tmp_dir_path / "lens_distortion_estimation_output.dat"
        )

        # run the lens distortion estimation executable
        subprocess.run(
            [
                str(lens_distortion_estimation_executable_path),
                str(median_image_file_path),
                str(undistorted_image_file_path),
                str(line_segment_points_file_path),
                str(lens_distortion_estimation_output_file_path),
            ]
        )

        # parse the distortion coefficients
        dist_coeffs = parse_dist_coeffs(lens_distortion_estimation_output_file_path)

        # save the distortion coefficients
        np.savetxt(str(output_dir_path / "dist_coeffs.txt"), dist_coeffs)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images_dir",
        type=lambda p: Path(p).resolve(strict=True),
        help="Path to directory containing images",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=lambda p: Path(p).resolve(strict=True),
        default=None,
        help="Path to directory where the output should be saved",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.images_dir, output_dir_path=args.output_dir)
