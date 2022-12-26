import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

from .line_detection import (
    KorniaSOLD2LineSegmentDetector,
    group_collinear_segments,
    merge_collinear_line_segment_groups,
)
from .intersections_detection import (
    detect_intersections,
    IntersectionType,
    INTERSECTIONS_3D_LOCATIONS,
)

from .visualization import (
    draw_line_segments,
    draw_intersections,
    draw_line_segments_groups,
)


def compute_field_mask(
    image,
    green_thresh=120,
    dilatation_size=3,
    dilation_iterations=3,
):
    # Convert to L*a*b
    image_lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)

    # Threshold to get only green pixels
    binarized_image = np.zeros(image.shape[:2], dtype=np.uint8)
    binarized_image[image_lab[:, :, 1] < green_thresh] = 255

    # Compute connected components
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binarized_image)

    # Filter out connected components that are too small
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > 1000:
            component_mask = (labels == i).astype(np.uint8) * 255
            mask = cv.bitwise_or(mask, component_mask)

    # Apply dilation to remove small holes
    element = cv.getStructuringElement(
        cv.MORPH_CROSS,
        (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        (dilatation_size, dilatation_size),
    )
    dilated_mask = cv.dilate(mask, element, iterations=dilation_iterations)

    return dilated_mask


def remove_line_segments_outside_field(line_segments, field_mask):
    res = []
    for l in line_segments:
        p1, p2 = l[0, :], l[1, :]
        p1_in_field = field_mask[int(p1[0]), int(p1[1])]
        p2_in_field = field_mask[int(p2[0]), int(p2[1])]

        if p1_in_field and p2_in_field:
            res.append(l)
    return np.array(res)


def compute_camera_extrinsics_from_field_lines(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    line_segments: np.ndarray,
):

    cv.imshow("Debug", image)
    cv.setWindowTitle("Debug", "Target image")
    cv.waitKey(1000)

    # Compute field mask
    field_mask = compute_field_mask(image)
    cv.imshow("Debug", field_mask)
    cv.setWindowTitle("Debug", "Field mask")
    cv.waitKey(1000)

    orig_line_segments_image = draw_line_segments(image, line_segments)
    cv.imshow("Debug", orig_line_segments_image)
    cv.setWindowTitle("Debug", "Line segments")
    cv.waitKey(1000)

    # Merge collinear line segments
    collinear_line_segments_groups = group_collinear_segments(line_segments)
    collinear_line_segments_groups_image = draw_line_segments_groups(
        image, line_segments, collinear_line_segments_groups
    )
    cv.imshow("Debug", collinear_line_segments_groups_image)
    cv.setWindowTitle("Debug", "Collinear line segments groups")
    cv.waitKey(1000)

    merged_line_segments = merge_collinear_line_segment_groups(
        line_segments, collinear_line_segments_groups
    )
    merged_line_segments_image = draw_line_segments(image, merged_line_segments)
    cv.imshow("Debug", merged_line_segments_image)
    cv.setWindowTitle("Debug", "Merged line segments")
    cv.waitKey(1000)

    # Filter out lines which are not inside the field mask
    merged_line_segments = remove_line_segments_outside_field(
        merged_line_segments, field_mask
    )
    filtered_line_segments_image = draw_line_segments(
        image, merged_line_segments, show_index=True
    )
    cv.imshow("Debug", filtered_line_segments_image)
    cv.setWindowTitle("Debug", "Filtered line segments")
    cv.waitKey(1000)

    # Detect intersections
    intersections, intersections_types, intersection_points_3d = detect_intersections(
        merged_line_segments,
    )
    intersections_image = draw_intersections(image, intersections, intersections_types)
    cv.imshow("Debug", intersections_image)
    cv.setWindowTitle("Debug", "Intersections")
    cv.waitKey(3000)

    # Estimate camera pose
    world_points = []
    image_points = []

    for idx, it in enumerate(intersections_types):
        if it == IntersectionType.UNKNOWN:
            continue
        world_points.append(intersection_points_3d[idx])
        intersection_point_rescaled = intersections[idx, 0, :]
        image_points.append(intersection_point_rescaled[::-1])

    world_points = np.array(world_points).astype(np.float32)
    image_points = np.array(image_points).astype(np.float32)

    retval, rvec, tvec = cv.solvePnP(
        world_points,
        image_points,
        camera_matrix,
        np.array([[0], [0], [0], [0], [0]], dtype=np.float32),
        flags=cv.SOLVEPNP_IPPE,
    )

    if retval != True:
        raise Exception("Could not estimate camera pose")

    # Refine camera pose with LM
    rvec, tvec = cv.solvePnPRefineLM(
        world_points,
        image_points,
        camera_matrix,
        np.array([[0], [0], [0], [0], [0]], dtype=np.float32),
        rvec,
        tvec,
    )

    # Show reprojected points
    test_points = []
    for p in INTERSECTIONS_3D_LOCATIONS.values():
        test_points.append(p)
    test_points = np.stack(test_points, axis=0)
    proj_test_points = cv.projectPoints(
        test_points,
        rvec,
        tvec,
        camera_matrix,
        np.array([[0], [0], [0], [0], [0]], dtype=np.float32),
    )[0]
    proj_test_points = np.squeeze(proj_test_points, axis=1)
    reprojected_points_image = np.copy(image)
    for p in proj_test_points:
        cv.circle(reprojected_points_image, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    cv.imshow("Debug", reprojected_points_image)
    cv.setWindowTitle("Debug", "Reprojected points")
    cv.waitKey(3000)

    cv.destroyAllWindows()

    return rvec, tvec


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Compute camera extrinsics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=lambda p: Path(p).resolve(),
        help="Path to the image to use for calibration",
        required=True,
    )

    parser.add_argument(
        "--camera-matrix",
        type=lambda p: Path(p).resolve(),
        help="Path to the camera matrix file",
        required=True,
    )

    parser.add_argument(
        "--lines",
        type=lambda p: Path(p).resolve(),
        help="Path to the lines file",
        default=None,
    )

    parser.add_argument(
        "--line-detection-downscaling-factor",
        type=int,
        default=2,
        help="Scale factor for resizing the image for line detection",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    image = cv.imread(str(args.image))
    h, w = image.shape[:2]
    camera_matrix = np.loadtxt(str(args.camera_matrix)).astype(np.float32)
    line_detection_downscaling_factor = args.line_detection_downscaling_factor

    if args.lines is not None:
        line_segments = np.load(str(args.lines))["line_seg"]
    else:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        line_detector = KorniaSOLD2LineSegmentDetector()

        resized_image_gray = cv.resize(
            image_gray,
            (
                w // line_detection_downscaling_factor,
                h // line_detection_downscaling_factor,
            ),
            cv.INTER_AREA,
        )

        line_segments = line_detector.detect_line_segments(resized_image_gray)

    # Scale line segments back to original image size
    line_segments *= line_detection_downscaling_factor

    rvec, tvec = compute_camera_extrinsics_from_field_lines(
        image,
        camera_matrix,
        line_segments,
    )
    exstrinsics_file_path = args.image.parent / "camera_extrinsics.npz"
    np.savez(exstrinsics_file_path, rvec=rvec, tvec=tvec)
