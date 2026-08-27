#!/usr/bin/env python3
"""Segment white-workpiece edges and fit an inscribed rotated rectangle.

YOLO-OBB limits the RGB search region. Canny edges, morphology and connected
components produce the detailed silhouette, then a large rotated rectangle is
searched entirely inside that silhouette. No HSV/gray threshold mode remains.

用于细化白色工件轮廓、计算内接旋转矩形
"""

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


WINDOW = "White edge segmentation and inscribed rectangle"


def image_message_to_bgr(message):
    if message.encoding not in ("bgr8", "rgb8"):
        raise ValueError("expected bgr8/rgb8, got {}".format(message.encoding))
    row_bytes = message.width * 3
    raw = np.frombuffer(message.data, dtype=np.uint8)
    rows = raw.reshape(message.height, message.step)
    image = rows[:, :row_bytes].reshape(message.height, message.width, 3).copy()
    if message.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def bgr_to_message(image, header):
    message = Image()
    message.header = header
    message.height, message.width = image.shape[:2]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = image.tobytes()
    return message


def mono_to_message(mask, header):
    message = Image()
    message.header = header
    message.height, message.width = mask.shape
    message.encoding = "mono8"
    message.is_bigendian = 0
    message.step = message.width
    message.data = mask.tobytes()
    return message


def scaled_polygon(points, scale, image_shape):
    points = np.asarray(points, dtype=np.float32)
    center = np.mean(points, axis=0, keepdims=True)
    result = center + float(scale) * (points - center)
    height, width = image_shape[:2]
    result[:, 0] = np.clip(result[:, 0], 0, width - 1)
    result[:, 1] = np.clip(result[:, 1], 0, height - 1)
    return np.rint(result).astype(np.int32)


def polygon_mask(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def fill_external_contour(mask, epsilon_fraction):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("white threshold produced no contour")
    contour = max(contours, key=cv2.contourArea)
    epsilon = float(epsilon_fraction) * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    return contour, filled


def select_best_component(binary, center, min_area):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("white threshold produced no connected region")
    cx, cy = float(center[0]), float(center[1])
    best_label = 0
    best_score = -np.inf
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        distance = float(np.linalg.norm(centroids[label] - [cx, cy]))
        # Prefer a large component close to the YOLO center.  This rejects
        # isolated table glare that also passes the white threshold.
        score = float(area) - 12.0 * distance
        column, row = int(round(cx)), int(round(cy))
        if 0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]:
            if labels[row, column] == label:
                score += float(area)
        if score > best_score:
            best_score = score
            best_label = label
    if best_label == 0:
        raise ValueError("no white component satisfies min_area")
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def largest_axis_aligned_rectangle(binary):
    """Return the largest all-foreground rectangle (inclusive coordinates)."""
    foreground = binary > 0
    height, width = foreground.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = None
    for row in range(height):
        histogram = np.where(foreground[row], histogram + 1, 0)
        stack = []
        for column in range(width + 1):
            current = int(histogram[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current:
                start_index, bar_height = stack.pop()
                area = bar_height * (column - start_index)
                if area > best_area:
                    best_area = area
                    best = (
                        start_index,
                        row - bar_height + 1,
                        column - 1,
                        row,
                    )
                start = start_index
            if current > 0 and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    if best is None:
        raise ValueError("no inscribed rectangle exists in segmented mask")
    return best


def contour_long_axis_angle(contour):
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    vectors = np.roll(box, -1, axis=0) - box
    vector = vectors[int(np.argmax(np.linalg.norm(vectors, axis=1)))]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def maximum_inscribed_rotated_rectangle(
    filled_mask, contour, angle_range_deg, angle_step_deg, margin_px
):
    """Search for a rotated rectangle completely contained by filled_mask."""
    x, y, width, height = cv2.boundingRect(contour)
    padding = int(margin_px) + 5
    side = int(np.ceil(np.hypot(width, height))) + 2 * padding
    side = max(side, width + 2 * padding, height + 2 * padding)
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = (
        filled_mask[y : y + height, x : x + width]
    )
    rotation_center = ((side - 1) * 0.5, (side - 1) * 0.5)
    base_angle = contour_long_axis_angle(contour)
    if angle_range_deg <= 0.0:
        offsets = [0.0]
    else:
        offsets = np.arange(
            -angle_range_deg,
            angle_range_deg + 0.5 * angle_step_deg,
            angle_step_deg,
        )

    best = None
    for offset in offsets:
        angle = base_angle + float(offset)
        matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        rotated = cv2.warpAffine(
            canvas,
            matrix,
            (side, side),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        left, top, right, bottom = largest_axis_aligned_rectangle(rotated)
        left += margin_px
        top += margin_px
        right -= margin_px
        bottom -= margin_px
        if right <= left or bottom <= top:
            continue
        area = float((right - left + 1) * (bottom - top + 1))
        if best is not None and area <= best[0]:
            continue
        rotated_corners = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float64,
        )
        inverse = cv2.invertAffineTransform(matrix)
        homogeneous = np.column_stack(
            (rotated_corners, np.ones(rotated_corners.shape[0]))
        )
        canvas_corners = homogeneous.dot(inverse.T)
        image_corners = canvas_corners.copy()
        image_corners[:, 0] += x - offset_x
        image_corners[:, 1] += y - offset_y
        best = (area, image_corners, angle)
    if best is None:
        raise ValueError("inscribed rectangle vanished after applying margin")
    return np.rint(best[1]).astype(np.int32), float(best[0]), float(best[2])


def segment_white(image, corners, args):
    search_polygon = scaled_polygon(corners, args.search_scale, image.shape)
    roi_mask = polygon_mask(image.shape, search_polygon)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (args.edge_blur_kernel, args.edge_blur_kernel), 0)
    edges = cv2.Canny(gray, args.canny_low, args.canny_high)
    edges = cv2.bitwise_and(edges, roi_mask)
    edge_close_size = max(1, int(args.edge_close_kernel) | 1)
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_close_size, edge_close_size)
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        edge_kernel,
        iterations=args.edge_close_iterations,
    )
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    edge_contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    white = np.zeros(edges.shape, dtype=np.uint8)
    center = np.mean(corners, axis=0)
    ranked = []
    for candidate in edge_contours:
        area = float(cv2.contourArea(candidate))
        if area < args.min_area:
            continue
        moments = cv2.moments(candidate)
        if abs(moments["m00"]) < 1e-9:
            continue
        candidate_center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        )
        distance = float(np.linalg.norm(candidate_center - center))
        ranked.append((area - 12.0 * distance, candidate))
    if not ranked:
        raise ValueError("Canny found no sufficiently large closed contour")
    best_edge_contour = max(ranked, key=lambda item: item[0])[1]
    cv2.drawContours(white, [best_edge_contour], -1, 255, cv2.FILLED)
    white = cv2.bitwise_and(white, roi_mask)

    close_size = max(1, int(args.close_kernel) | 1)
    open_size = max(1, int(args.open_kernel) | 1)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        iterations=args.close_iterations,
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    component = select_best_component(white, np.mean(corners, axis=0), args.min_area)
    contour, filled = fill_external_contour(component, args.contour_epsilon)
    if cv2.contourArea(contour) < args.min_area:
        raise ValueError("final white contour is smaller than min_area")
    rectangle, rectangle_area, rectangle_angle = maximum_inscribed_rotated_rectangle(
        filled,
        contour,
        args.inscribed_angle_range_deg,
        args.inscribed_angle_step_deg,
        args.inscribed_margin_px,
    )
    return (
        contour,
        rectangle,
        filled,
        search_polygon,
        rectangle_area,
        rectangle_angle,
    )


class WhiteEdgeSegmenter:
    def __init__(self):
        script_dir = Path(__file__).resolve().parent
        default_model = script_dir / "models" / "white_box.pt"
        model_path = Path(rospy.get_param("~model", str(default_model))).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError("model not found: {}".format(model_path))

        self.confidence = float(rospy.get_param("~confidence", 0.45))
        self.iou = float(rospy.get_param("~iou", 0.50))
        self.imgsz = int(rospy.get_param("~imgsz", 960))
        self.device = str(rospy.get_param("~device", "0"))
        self.search_scale = float(rospy.get_param("~search_scale", 1.08))
        self.canny_low = int(rospy.get_param("~canny_low", 35))
        self.canny_high = int(rospy.get_param("~canny_high", 110))
        self.edge_blur_kernel = int(rospy.get_param("~edge_blur_kernel", 5)) | 1
        self.edge_close_kernel = int(rospy.get_param("~edge_close_kernel", 9))
        self.edge_close_iterations = int(
            rospy.get_param("~edge_close_iterations", 2)
        )
        self.min_area = int(rospy.get_param("~min_area", 1200))
        self.close_kernel = int(rospy.get_param("~close_kernel", 11))
        self.close_iterations = int(rospy.get_param("~close_iterations", 2))
        self.open_kernel = int(rospy.get_param("~open_kernel", 3))
        self.contour_epsilon = float(rospy.get_param("~contour_epsilon", 0.002))
        self.inscribed_angle_range_deg = float(
            rospy.get_param("~inscribed_angle_range_deg", 2.0)
        )
        self.inscribed_angle_step_deg = float(
            rospy.get_param("~inscribed_angle_step_deg", 1.0)
        )
        self.inscribed_margin_px = int(rospy.get_param("~inscribed_margin_px", 3))
        self.show_gui = bool(rospy.get_param("~show_gui", True))
        self.save_dir = Path(
            rospy.get_param(
                "~save_dir", str(script_dir.parents[2] / "white_threshold_result")
            )
        ).expanduser()
        if self.search_scale < 1.0:
            raise ValueError("search_scale must be >= 1")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("require 0 <= canny_low < canny_high <= 255")
        if self.edge_blur_kernel < 3:
            raise ValueError("edge_blur_kernel must be at least 3")
        if self.edge_close_kernel < 1 or self.edge_close_iterations < 1:
            raise ValueError("edge close parameters must be positive")
        if self.min_area <= 0:
            raise ValueError("min_area must be positive")
        if self.inscribed_angle_range_deg < 0.0:
            raise ValueError("inscribed_angle_range_deg must be non-negative")
        if self.inscribed_angle_step_deg <= 0.0:
            raise ValueError("inscribed_angle_step_deg must be positive")
        if self.inscribed_margin_px < 0:
            raise ValueError("inscribed_margin_px must be non-negative")

        self.model = YOLO(str(model_path))
        self.lock = threading.Lock()
        self.latest = None
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.subscriber = rospy.Subscriber(
            image_topic, Image, self.callback, queue_size=1, buff_size=2**24
        )
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("~mask", Image, queue_size=1)
        self.result_pub = rospy.Publisher("~result", String, queue_size=1)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.show_gui:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        rospy.loginfo("White YOLO model: %s", model_path)
        rospy.loginfo("Color topic: %s", image_topic)
        rospy.loginfo(
            "White segmentation: EDGE Canny=%d/%d close=%dx%d",
            self.canny_low,
            self.canny_high,
            self.edge_close_kernel,
            self.edge_close_iterations,
        )

    def callback(self, message):
        with self.lock:
            self.latest = message

    def process_latest(self):
        with self.lock:
            message, self.latest = self.latest, None
        if message is None:
            rospy.logwarn_throttle(3.0, "Waiting for color image")
            if self.show_gui:
                cv2.waitKey(1)
            return

        image = image_message_to_bgr(message)
        prediction = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        debug = image.copy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        objects = []
        if prediction.obb is not None:
            corner_sets = prediction.obb.xyxyxyxy.cpu().numpy()
            scores = prediction.obb.conf.cpu().numpy()
            for index, (corners, confidence) in enumerate(zip(corner_sets, scores)):
                yolo_polygon = np.rint(corners).astype(np.int32)
                cv2.polylines(debug, [yolo_polygon], True, (0, 255, 255), 2)
                try:
                    (
                        contour,
                        rectangle,
                        mask,
                        _,
                        rectangle_area,
                        rectangle_angle,
                    ) = segment_white(image, corners, self)
                except ValueError as error:
                    center = np.rint(np.mean(corners, axis=0)).astype(int)
                    cv2.putText(
                        debug,
                        "#{} WHITE FAILED".format(index),
                        tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    rospy.logwarn_throttle(2.0, "White segmentation failed: %s", error)
                    continue

                combined_mask = cv2.bitwise_or(combined_mask, mask)
                overlay = np.zeros_like(debug)
                overlay[mask > 0] = (0, 180, 0)
                debug = cv2.addWeighted(debug, 1.0, overlay, 0.25, 0.0)
                cv2.drawContours(debug, [contour], -1, (255, 0, 255), 3)
                cv2.polylines(debug, [rectangle], True, (255, 0, 0), 2)
                center = np.rint(np.mean(corners, axis=0)).astype(int)
                area = float(cv2.contourArea(contour))
                cv2.putText(
                    debug,
                    "#{} conf={:.2f} white_area={:.0f}".format(
                        index, confidence, area
                    ),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 0, 255),
                    2,
                )
                objects.append(
                    {
                        "id": index,
                        "confidence": round(float(confidence), 5),
                        "yolo_obb_px": np.round(corners, 2).tolist(),
                        "white_contour_px": contour.reshape(-1, 2).astype(int).tolist(),
                        "inscribed_rectangle_px": rectangle.astype(int).tolist(),
                        "fitted_rectangle_px": rectangle.astype(int).tolist(),
                        "inscribed_rectangle_area_px": round(rectangle_area, 1),
                        "inscribed_rectangle_angle_deg": round(rectangle_angle, 3),
                        "white_area_px": round(area, 1),
                    }
                )

        cv2.putText(
            debug,
            "yellow=YOLO  magenta=edge contour  blue=inscribed rectangle",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),
            2,
        )
        self.debug_pub.publish(bgr_to_message(debug, message.header))
        self.mask_pub.publish(mono_to_message(combined_mask, message.header))
        self.result_pub.publish(
            String(
                data=json.dumps(
                    {
                        "count": len(objects),
                        "objects": objects,
                        "stamp": message.header.stamp.to_sec(),
                    }
                )
            )
        )
        cv2.imwrite(str(self.save_dir / "latest_white_edge.png"), debug)
        cv2.imwrite(str(self.save_dir / "latest_white_mask.png"), combined_mask)
        rospy.loginfo_throttle(1.0, "White-threshold segmented %d object(s)", len(objects))
        if self.show_gui:
            cv2.imshow(WINDOW, debug)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("q pressed")


def main():
    rospy.init_node("white_edge_segmenter")
    node = WhiteEdgeSegmenter()
    rospy.on_shutdown(cv2.destroyAllWindows)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        try:
            node.process_latest()
        except (cv2.error, RuntimeError, ValueError) as error:
            rospy.logerr_throttle(2.0, "White threshold error: %s", error)
        rate.sleep()


if __name__ == "__main__":
    main()
