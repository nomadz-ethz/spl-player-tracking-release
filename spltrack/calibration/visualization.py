import numpy as np
import cv2 as cv


from .intersections_detection import IntersectionType

INTERSECTIONS_COLOR_MAP = {
    IntersectionType.UNKNOWN: (0, 0, 0),
    IntersectionType.TOP_LEFT_CORNER: (255, 0, 0),
    IntersectionType.TOP_RIGHT_CORNER: (0, 255, 0),
    IntersectionType.LEFT_GOAL_BOX_TOP_CORNER: (0, 0, 255),
    IntersectionType.LEFT_GOAL_BOX_BOTTOM_CORNER: (255, 255, 0),
    IntersectionType.RIGHT_GOAL_BOX_TOP_CORNER: (255, 0, 255),
    IntersectionType.RIGHT_GOAL_BOX_BOTTOM_CORNER: (0, 255, 255),
    IntersectionType.TOP_MIDFIELD_T_INTERSECTION: (255, 0, 0),
    IntersectionType.BOTTOM_MIDFIELD_T_INTERSECTION: (0, 255, 0),
}


def draw_line_segments(image, line_segments, show_index=False):
    res = np.copy(image)
    for idx, l in enumerate(line_segments):
        y1, x1 = l[0]
        y2, x2 = l[1]
        color = tuple([int(c) for c in np.random.randint(0, 255, size=(3,), dtype=int)])
        cv.line(res, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        if show_index:
            y_m = (y1 + y2) / 2
            x_m = (x1 + x2) / 2
            cv.putText(
                res,
                str(idx),
                (int(x_m), int(y_m)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
    return res


def draw_line_segments_groups(
    image, line_segments, line_segments_groups, show_index=False
):
    res = np.copy(image)
    for group in line_segments_groups:
        color = tuple([int(c) for c in np.random.randint(0, 255, size=(3,), dtype=int)])
        for line_idx in group:
            y1, x1 = line_segments[line_idx, 0]
            y2, x2 = line_segments[line_idx, 1]
            cv.line(res, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            if show_index:
                y_m = (y1 + y2) / 2
                x_m = (x1 + x2) / 2
                cv.putText(
                    res,
                    str(line_idx),
                    (int(x_m), int(y_m)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
    return res


def draw_intersections(image, intersections, intersection_types):
    res = np.copy(image)
    for (p, d1, d2), i_type in zip(intersections, intersection_types):
        color = INTERSECTIONS_COLOR_MAP[i_type]
        y0, x0 = p
        cv.circle(res, (int(x0), int(y0)), 5, (0, 0, 255), -1)
        color = tuple([int(c) for c in np.random.randint(0, 255, size=(3,), dtype=int)])
        y1, x1 = p + 50 * d1
        cv.arrowedLine(res, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        y2, x2 = p + 50 * d2
        cv.arrowedLine(res, (int(x0), int(y0)), (int(x2), int(y2)), color, 2)
        cv.putText(
            res, i_type.name, (int(x0), int(y0)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
    return res
