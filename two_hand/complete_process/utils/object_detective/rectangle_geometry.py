"""Shared rotated-rectangle geometry for object detectors.
   被深度检测器导入，用来从二值掩膜中提取旋转矩形
"""

import cv2
import numpy as np


def order_corners(points):
    """Order four image points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = np.argmin(points[:, 0] + points[:, 1])
    points = np.roll(points, -start, axis=0)
    if points[1, 0] < points[-1, 0]:
        points = points[[0, 3, 2, 1]]
    return points


def detect_rectangles(mask, min_area, max_objects):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        rectangle = cv2.minAreaRect(contour)
        center, (width, height), _ = rectangle
        short_side, long_side = sorted((width, height))
        if short_side < 2.0:
            continue
        aspect = long_side / short_side
        rectangle_area = width * height
        rectangularity = area / rectangle_area if rectangle_area else 0.0

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area else 0.0

        corners = order_corners(cv2.boxPoints(rectangle))
        edges = np.roll(corners, -1, axis=0) - corners
        long_edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        angle = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
        if angle >= 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        detections.append({
            "area": area,
            "aspect": aspect,
            "rectangularity": rectangularity,
            "solidity": solidity,
            "center": np.asarray(center, dtype=np.float32),
            "angle": angle,
            "corners": corners,
        })

    detections.sort(key=lambda item: item["area"], reverse=True)
    if max_objects > 0:
        detections = detections[:max_objects]
    return detections
