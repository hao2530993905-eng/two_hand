#!/usr/bin/env python3
"""Validate and install ost.yaml from camera_calibration's saved archive."""

import argparse
import datetime
import shutil
import sys
import tarfile
from pathlib import Path

import yaml


REQUIRED_LENGTHS = {
    "camera_matrix": 9,
    "rectification_matrix": 9,
    "projection_matrix": 12,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", default="/tmp/calibrationdata.tar.gz")
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    return parser.parse_args()


def validate(document, width, height):
    if int(document.get("image_width", -1)) != width or int(document.get("image_height", -1)) != height:
        raise ValueError("calibration resolution does not match the requested stream")
    if document.get("distortion_model") != "plumb_bob":
        raise ValueError("expected a plumb_bob color-camera model")
    for key, length in REQUIRED_LENGTHS.items():
        if len(document.get(key, {}).get("data", [])) != length:
            raise ValueError("{} must contain {} values".format(key, length))
    distortion = document.get("distortion_coefficients", {}).get("data", [])
    if len(distortion) < 4:
        raise ValueError("distortion_coefficients must contain at least four values")


def main():
    args = parse_args()
    archive = Path(args.archive).expanduser().resolve()
    package = Path(__file__).resolve().parents[1]
    output = package / "config" / "d435_color_intrinsics.yaml"
    result_dir = package.parents[2] / "calibration_results"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with tarfile.open(str(archive), "r:gz") as bundle:
        member = next((item for item in bundle.getmembers() if Path(item.name).name == "ost.yaml"), None)
        if member is None or not member.isfile():
            raise ValueError("archive does not contain ost.yaml")
        stream = bundle.extractfile(member)
        document = yaml.safe_load(stream.read().decode("utf-8"))
    validate(document, args.expected_width, args.expected_height)
    document["camera_name"] = "d435_color"
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = result_dir / "d435_135622076024_intrinsics_{}.tar.gz".format(stamp)
    shutil.copy2(str(archive), str(archived))
    print("Installed: {}".format(output))
    print("Archived raw samples: {}".format(archived))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, yaml.YAMLError, tarfile.TarError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)

