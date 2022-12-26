import numpy as np
import cv2 as cv


def get_distortion_lut(max_dist: float, dist_coeffs: np.ndarray) -> np.ndarray:
    lut_size = int(max_dist + 2)
    lut = np.zeros(lut_size)
    b = np.concatenate([[0], dist_coeffs])[::-1]
    root = root2 = 1.0
    b2 = np.polyder(np.concatenate([dist_coeffs[::-1], [0]]))
    for dist in np.arange(1, lut_size):
        b[-1] = -float(dist)
        for k in range(10000):
            pol_eval = np.polyval(b, root)
            pol_der = np.polyval(b2, root)
            if np.abs(pol_der) < 1e-14:
                break
            root2 = root - pol_eval / pol_der
            if np.abs(root - root2) < np.abs(root) * 1e-8:
                root = root2
                break
            root = root2
        lut[dist] = root / float(dist)
    lut[0] = lut[1]
    return lut


def get_max_distance_from_point(width, height, point):
    corners = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )
    return np.max(np.linalg.norm(corners - point, axis=1))


def undistort_image(
    image: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]
    principal_point = np.array([w / 2, h / 2])
    # Compute distortion model lut
    max_distance = get_max_distance_from_point(w, h, principal_point)
    dm_lut = get_distortion_lut(max_distance, dist_coeffs)

    # Compute distortion factor for every image point
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    image_points = np.stack([X, Y], axis=2)
    image_points_flat = image_points.reshape((-1, 2))
    dists = np.linalg.norm(image_points_flat - principal_point, axis=1)
    dists_ind = np.round(dists).astype(np.int64)
    df = dm_lut[dists_ind] + (dists - dists_ind) * (
        dm_lut[dists_ind + 1] - dm_lut[dists_ind]
    )

    # Undistort
    image_points_undistorted_flat = principal_point + np.multiply(
        image_points_flat - principal_point, df[:, None]
    )
    image_points_undistorted = np.reshape(
        image_points_undistorted_flat, (h, w, 2)
    ).astype(np.float32)

    # Warp
    map_x, map_y = image_points_undistorted[:, :, 0], image_points_undistorted[:, :, 1]
    undistorted_image = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR)
    return undistorted_image


class RadialDistortionCorrector:
    def __init__(self, width, height, dist_coeffs):
        self.width = width
        self.height = height
        self.principal_point = np.array([width / 2, height / 2])
        self.dist_coeffs = dist_coeffs

        max_dist_from_pc = get_max_distance_from_point(
            self.width,
            self.height,
            self.principal_point,
        )
        self.dm_lut = get_distortion_lut(
            max_dist_from_pc,
            self.dist_coeffs,
        )

        # Compute distortion factor for every image point
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        image_points = np.stack([X, Y], axis=2)
        image_points_flat = image_points.reshape((-1, 2))
        dists = np.linalg.norm(image_points_flat - self.principal_point, axis=1)
        dists_ind = np.round(dists).astype(np.int64)
        df = self.dm_lut[dists_ind] + (dists - dists_ind) * (
            self.dm_lut[dists_ind + 1] - self.dm_lut[dists_ind]
        )

        # Undistort
        image_points_undistorted_flat = self.principal_point + np.multiply(
            image_points_flat - self.principal_point, df[:, None]
        )
        self.image_points_undistorted = np.reshape(
            image_points_undistorted_flat, (self.height, self.width, 2)
        ).astype(np.float32)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        undistorted_image = cv.remap(
            image,
            self.image_points_undistorted[:, :, 0],
            self.image_points_undistorted[:, :, 1],
            interpolation=cv.INTER_LINEAR,
        )
        return undistorted_image


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        "Undistort images using a radial distortion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img_list",
        type=lambda p: Path(p).resolve(),
        help="Text file with list of images to undistort",
        required=True,
    )

    parser.add_argument(
        "--dist-coeffs",
        type=lambda p: Path(p).resolve(),
        default=Path(__file__).parent.joinpath("radial_distortion_params.txt"),
    )

    args = parser.parse_args()

    with open(args.img_list, "r") as f:
        images = [Path(line.strip()) for line in f]

    dist_coeffs = np.loadtxt(args.dist_coeffs)

    for image_file_path in images:
        image = cv.imread(str(image_file_path))
        undistorted_image = undistort_image(image, dist_coeffs)
        cv.imwrite(
            str(image_file_path.with_suffix(".undistorted.jpg")), undistorted_image
        )
