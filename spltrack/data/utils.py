from pathlib import Path
from typing import List, Optional

import numpy as np


def bbox_from_points(xtl, ytl, xbr, ybr, mode="x0y0x1y1"):

    if mode == "x0y0x1y1":
        return np.array([xtl, ytl, xbr, ybr])
    elif mode == "xywh":
        return np.array([xtl, ytl, xbr - xtl, ybr - ytl])
    else:
        raise ValueError(f"Invalid box mode: {mode}")


def find_sequence_dirs(dataset_dir: Path, sequences: Optional[List[str]] = None):
    data_dirs = []
    for p in dataset_dir.iterdir():
        if p.joinpath("images").is_dir() and p.joinpath("gc").is_dir():
            if sequences is not None and p.name not in sequences:
                continue
            data_dirs.append(p)

    return data_dirs
