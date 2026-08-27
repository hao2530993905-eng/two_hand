#!/usr/bin/env python3
"""Generate the project's dimensioned A4 ChArUco target as PNG and PDF."""

from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


COLS = 7
ROWS = 10
SQUARE_MM = 25.0
MARKER_MM = 18.0
PIXELS_PER_MM = 20
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "calibration_targets"


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(
        COLS, ROWS, SQUARE_MM, MARKER_MM, dictionary
    )
    width_px = int(COLS * SQUARE_MM * PIXELS_PER_MM)
    height_px = int(ROWS * SQUARE_MM * PIXELS_PER_MM)
    image = board.draw((width_px, height_px), marginSize=0, borderBits=1)

    # Refuse to save a target that this host's detector cannot read back.
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )
    expected = (COLS - 1) * (ROWS - 1)
    if len(ids) != (COLS * ROWS) // 2 or int(count) != expected:
        raise RuntimeError("generated target failed marker/corner validation")
    if charuco_corners is None or charuco_ids is None:
        raise RuntimeError("generated target has no ChArUco corners")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "charuco_5x5_250_7x10_A4.png"
    pdf_path = OUTPUT_DIR / "charuco_5x5_250_7x10_A4.pdf"
    if not cv2.imwrite(str(png_path), image):
        raise RuntimeError("failed to write PNG")

    page_width, page_height = A4
    board_width = COLS * SQUARE_MM * mm
    board_height = ROWS * SQUARE_MM * mm
    left = (page_width - board_width) / 2.0
    bottom = (page_height - board_height) / 2.0
    document = canvas.Canvas(str(pdf_path), pagesize=A4, pageCompression=1)
    document.drawImage(
        str(png_path), left, bottom, width=board_width, height=board_height,
        preserveAspectRatio=True, mask=None,
    )
    # A 100 mm scale bar outside the board catches print-dialog scaling.
    bar_y = 10.0 * mm
    bar_left = (page_width - 100.0 * mm) / 2.0
    document.setLineWidth(0.35 * mm)
    document.line(bar_left, bar_y, bar_left + 100.0 * mm, bar_y)
    document.line(bar_left, bar_y - 2.0 * mm, bar_left, bar_y + 2.0 * mm)
    document.line(
        bar_left + 100.0 * mm, bar_y - 2.0 * mm,
        bar_left + 100.0 * mm, bar_y + 2.0 * mm,
    )
    document.setFont("Helvetica", 8)
    document.drawCentredString(page_width / 2.0, 5.0 * mm, "100 mm print check")
    document.showPage()
    document.save()
    print(png_path)
    print(pdf_path)
    print("validated markers={} charuco_corners={}".format(len(ids), int(count)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

