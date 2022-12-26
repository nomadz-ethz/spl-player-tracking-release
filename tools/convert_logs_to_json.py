#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path


def convert(path):
    assert path.name.endswith(".raw.json")
    with open(path, "r") as f:
        o = json.load(f)

    tc = [
        {
            "timestamp": msg["time"],
            "teamNum": msg["data"]["message"][6],
            "playerNum": msg["data"]["message"][5],
            "pose": list(
                struct.unpack(
                    "fff", bytes((b % 256) for b in msg["data"]["message"][8:20])
                )
            ),
            "fallen": msg["data"]["message"][7] == 1,
        }
        for msg in o
        if msg["data"]["message"]
    ]
    g = [
        {
            "timestamp": msg["time"],
            **{k: v for k, v in msg["data"].items() if k not in ["message"]},
        }
        for msg in o
        if not msg["data"]["message"]
    ]

    tc_path = path.parent / path.name.replace(".raw.json", ".tc.json")
    g_path = path.parent / path.name.replace(".raw.json", ".g.json")
    print(f"Writing to\n\tTC: {tc_path}\n\tGC: {g_path}")
    with open(tc_path, "w") as f2:
        json.dump(tc, f2)
    with open(g_path, "w") as f2:
        json.dump(g, f2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RoboCup SPL *.raw.json files to *.tc.json & *.g.json files (overwrites any existing output!)"
    )
    parser.add_argument("raw", type=Path, nargs="+", help="path(s) to *.raw.json")
    args = parser.parse_args()

    for path in args.raw:
        convert(path)
