#!/usr/bin/env python3
"""Draw boxes on D435 frames and save a one-class YOLO dataset."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import rosgraph
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


Box = Tuple[int, int, int, int]
Annotation = Union[Box, Sequence[Sequence[float]], np.ndarray]
WINDOW_NAME = "Manual YOLO annotation"
GREEN = (0, 220, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def normalized_box(box: Sequence[int], width: int, height: int) -> Box:
    """Clamp and order an image-space rectangle."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = sorted((
        max(0, min(x1, width - 1)),
        max(0, min(x2, width - 1)),
    ))
    y1, y2 = sorted((
        max(0, min(y1, height - 1)),
        max(0, min(y2, height - 1)),
    ))
    return x1, y1, x2, y2


def annotation_corners(
    annotation: Annotation, width: int, height: int
) -> np.ndarray:
    """Return four clipped, consecutive image-space corners."""
    values = np.asarray(annotation, dtype=np.float32)
    if values.shape == (4,):
        x1, y1, x2, y2 = normalized_box(values, width, height)
        values = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
    elif values.shape != (4, 2):
        raise ValueError("annotation must be a box or four 2-D corners")
    values[:, 0] = np.clip(values[:, 0], 0, width - 1)
    values[:, 1] = np.clip(values[:, 1], 0, height - 1)
    if abs(float(cv2.contourArea(values))) < 1.0:
        raise ValueError("annotation has zero area")
    return values


def rotated_rectangle_corners(
    axis_start: Sequence[float],
    axis_end: Sequence[float],
    width_point: Sequence[float],
) -> np.ndarray:
    """Build corners from a centerline and a point controlling half-width."""
    start = np.asarray(axis_start, dtype=np.float32)
    end = np.asarray(axis_end, dtype=np.float32)
    point = np.asarray(width_point, dtype=np.float32)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1.0:
        raise ValueError("rotated rectangle axis is too short")
    normal = np.array((-axis[1], axis[0]), dtype=np.float32) / length
    midpoint = (start + end) * 0.5
    signed_half_width = float(np.dot(point - midpoint, normal))
    if abs(signed_half_width) < 0.5:
        raise ValueError("rotated rectangle width is too small")
    offset = normal * signed_half_width
    return np.asarray(
        [start + offset, end + offset, end - offset, start - offset],
        dtype=np.float32,
    )


def yolo_obb_line(
    class_id: int, annotation: Annotation, width: int, height: int
) -> str:
    """Encode four consecutive rectangle corners as YOLO-OBB."""
    corners = annotation_corners(annotation, width, height)
    normalized = corners.copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    return "{} {}".format(
        int(class_id),
        " ".join("{:.6f}".format(value) for value in normalized.reshape(-1)),
    )


def yolo_detect_line(
    class_id: int, annotation: Annotation, width: int, height: int
) -> str:
    """Encode a drag box in standard YOLO center-x/center-y/width/height format."""
    corners = annotation_corners(annotation, width, height)
    x1, y1 = np.min(corners, axis=0)
    x2, y2 = np.max(corners, axis=0)
    return "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
        int(class_id),
        (x1 + x2) * 0.5 / width,
        (y1 + y2) * 0.5 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Annotation],
    active_box: Optional[Annotation] = None,
) -> np.ndarray:
    preview = image.copy()
    height, width = image.shape[:2]
    for index, annotation in enumerate(boxes):
        corners = annotation_corners(annotation, width, height)
        integer_corners = np.rint(corners).astype(np.int32)
        cv2.polylines(preview, [integer_corners], True, GREEN, 2)
        label_x, label_y = integer_corners[0]
        cv2.putText(
            preview, "#{}".format(index + 1),
            (int(label_x), max(18, int(label_y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2,
        )
    if active_box is not None:
        corners = annotation_corners(active_box, width, height)
        cv2.polylines(
            preview, [np.rint(corners).astype(np.int32)], True, YELLOW, 2
        )
    return preview


class DatasetWriter:
    """Save images, labels and previews while tracking session-local undo."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        class_name: str = "white_rectangle",
        class_id: int = 0,
        label_format: str = "obb",
        jpeg_quality: int = 95,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        if class_id != 0:
            raise ValueError("this one-class annotator requires class_id=0")
        if label_format not in ("obb", "detect"):
            raise ValueError("label_format must be obb or detect")
        if not class_name.strip():
            raise ValueError("class_name must not be empty")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = split
        self.class_name = class_name.strip()
        self.class_id = class_id
        self.label_format = label_format
        self.jpeg_quality = jpeg_quality
        self.saved_samples: List[Tuple[Path, Path, Path]] = []
        self.initialize()

    def initialize(self) -> None:
        for category in ("images", "labels", "previews"):
            (self.dataset_dir / category / self.split).mkdir(
                parents=True, exist_ok=True
            )
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            yaml_path.write_text(
                "path: {}\n"
                "train: images/train\n"
                "val: images/val\n\n"
                "names:\n"
                "  0: {}\n".format(self.dataset_dir, self.class_name),
                encoding="utf-8",
            )

    def sample_paths(self) -> Tuple[Path, Path, Path]:
        name = "manual_{}_{}".format(
            int(time.time() * 1000), time.time_ns() % 1_000_000
        )
        return (
            self.dataset_dir / "images" / self.split / (name + ".jpg"),
            self.dataset_dir / "labels" / self.split / (name + ".txt"),
            self.dataset_dir / "previews" / self.split / (name + ".jpg"),
        )

    def save(
        self,
        image: np.ndarray,
        boxes: Sequence[Annotation],
        negative: bool = False,
    ) -> Tuple[Path, int]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected a BGR image")
        if not negative and not boxes:
            raise ValueError("positive samples require at least one box")

        height, width = image.shape[:2]
        checked = [
            annotation_corners(annotation, width, height)
            for annotation in boxes
        ]
        if negative:
            checked = []

        encoder = (
            yolo_obb_line if self.label_format == "obb" else yolo_detect_line
        )
        labels = [
            encoder(self.class_id, box, width, height) for box in checked
        ]
        image_path, label_path, preview_path = self.sample_paths()
        preview = draw_boxes(image, checked)
        if negative:
            cv2.putText(
                preview, "NEGATIVE (empty label)", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2,
            )

        written: List[Path] = []
        try:
            if not cv2.imwrite(
                str(image_path), image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            ):
                raise OSError("cannot save image: {}".format(image_path))
            written.append(image_path)
            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            written.append(label_path)
            if not cv2.imwrite(
                str(preview_path), preview,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            ):
                raise OSError("cannot save preview: {}".format(preview_path))
            written.append(preview_path)
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise

        self.saved_samples.append((image_path, label_path, preview_path))
        return image_path, len(labels)

    def undo(self) -> Optional[Path]:
        if not self.saved_samples:
            return None
        paths = self.saved_samples.pop()
        for path in paths:
            path.unlink(missing_ok=True)
        return paths[0]


class ManagedProcess:
    """A process group started by this tool and stopped on tool exit."""

    def __init__(self, command: Sequence[str], name: str) -> None:
        self.name = name
        self.process = subprocess.Popen(list(command), start_new_session=True)

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=8.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is not None:
                return
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)


def wait_for_master(timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if rosgraph.is_master_online():
            return True
        time.sleep(0.2)
    return False


def topic_is_published(topic: str) -> bool:
    try:
        return any(name == topic for name, _ in rospy.get_published_topics())
    except rospy.ROSException:
        return False


class ManualAnnotator:
    def __init__(self, args: argparse.Namespace, writer: DatasetWriter) -> None:
        self.args = args
        self.writer = writer
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frozen_frame: Optional[np.ndarray] = None
        self.boxes: List[np.ndarray] = []
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.pending_axis: Optional[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = None
        self.width_current: Optional[Tuple[int, int]] = None
        self.status = "Waiting for D435 image ..."
        self.status_color = YELLOW
        self.subscriber = rospy.Subscriber(
            args.topic, Image, self.image_callback,
            queue_size=1, buff_size=2 ** 24,
        )

    def image_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as error:
            self.set_status("CvBridge error: {}".format(error), RED)
            return
        with self.frame_lock:
            self.latest_frame = image.copy()

    def live_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return (
                None if self.latest_frame is None else self.latest_frame.copy()
            )

    def set_status(self, text: str, color: Tuple[int, int, int]) -> None:
        self.status = text
        self.status_color = color

    def freeze(self) -> bool:
        frame = self.live_frame()
        if frame is None:
            self.set_status("No camera frame yet", RED)
            return False
        self.frozen_frame = frame
        self.boxes = []
        self.drag_start = None
        self.drag_current = None
        self.pending_axis = None
        self.width_current = None
        if self.args.box_mode == "rotated":
            message = "FROZEN: drag along long axis, move sideways, click"
        else:
            message = "FROZEN: drag boxes, then press S"
        self.set_status(message, YELLOW)
        return True

    def resume(self) -> None:
        self.frozen_frame = None
        self.boxes = []
        self.drag_start = None
        self.drag_current = None
        self.pending_axis = None
        self.width_current = None
        self.set_status("LIVE: drag to freeze and start a box", GREEN)

    def clamp_point(self, x: int, y: int) -> Tuple[int, int]:
        if self.frozen_frame is None:
            return x, y
        height, width = self.frozen_frame.shape[:2]
        return max(0, min(x, width - 1)), max(0, min(y, height - 1))

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, userdata: object
    ) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frozen_frame is None and not self.freeze():
                return
            if self.args.box_mode == "rotated" and self.pending_axis is not None:
                point = self.clamp_point(x, y)
                try:
                    corners = rotated_rectangle_corners(
                        self.pending_axis[0], self.pending_axis[1], point
                    )
                    height, width = self.frozen_frame.shape[:2]
                    corners = annotation_corners(corners, width, height)
                except ValueError as error:
                    self.set_status(str(error), RED)
                    return
                short_edges = np.roll(corners, -1, axis=0) - corners
                edge_lengths = np.linalg.norm(short_edges, axis=1)
                if float(np.min(edge_lengths)) < self.args.min_box_size:
                    self.set_status(
                        "Width is smaller than {} px; move farther sideways".format(
                            self.args.min_box_size
                        ),
                        RED,
                    )
                    return
                self.boxes.append(corners)
                self.pending_axis = None
                self.width_current = None
                self.set_status(
                    "{} rotated box(es): draw more or press S".format(
                        len(self.boxes)
                    ),
                    YELLOW,
                )
                return
            self.drag_start = self.clamp_point(x, y)
            self.drag_current = self.drag_start
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                self.drag_current = self.clamp_point(x, y)
            elif self.pending_axis is not None:
                self.width_current = self.clamp_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            end = self.clamp_point(x, y)
            height, width = self.frozen_frame.shape[:2]
            start = self.drag_start
            self.drag_start = None
            self.drag_current = None
            if float(np.linalg.norm(
                np.asarray(end, dtype=np.float32)
                - np.asarray(start, dtype=np.float32)
            )) < self.args.min_box_size:
                self.set_status(
                    "Axis discarded: shorter than {} px".format(
                        self.args.min_box_size
                    ),
                    RED,
                )
                return
            if self.args.box_mode == "rotated":
                self.pending_axis = (start, end)
                self.width_current = end
                self.set_status(
                    "Move sideways to set width, then left-click to confirm",
                    YELLOW,
                )
                return
            box = normalized_box((*start, *end), width, height)
            if min(box[2] - box[0], box[3] - box[1]) < self.args.min_box_size:
                self.set_status(
                    "Box discarded: side shorter than {} px".format(
                        self.args.min_box_size
                    ),
                    RED,
                )
                return
            self.boxes.append(annotation_corners(box, width, height))
            self.set_status(
                "{} box(es): drag more or press S".format(len(self.boxes)),
                YELLOW,
            )
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pending_axis is not None or self.drag_start is not None:
                self.pending_axis = None
                self.width_current = None
                self.drag_start = None
                self.drag_current = None
                self.set_status("Cancelled unfinished box", YELLOW)
                return
            if not self.boxes:
                return
            px, py = self.clamp_point(x, y)
            for index in range(len(self.boxes) - 1, -1, -1):
                polygon = self.boxes[index].astype(np.float32)
                if cv2.pointPolygonTest(polygon, (px, py), False) >= 0:
                    del self.boxes[index]
                    self.set_status(
                        "Removed box; {} remain".format(len(self.boxes)), YELLOW
                    )
                    break

    def active_box(self) -> Optional[np.ndarray]:
        if self.frozen_frame is None:
            return None
        height, width = self.frozen_frame.shape[:2]
        if self.pending_axis is not None and self.width_current is not None:
            try:
                corners = rotated_rectangle_corners(
                    self.pending_axis[0],
                    self.pending_axis[1],
                    self.width_current,
                )
                return annotation_corners(corners, width, height)
            except ValueError:
                return None
        if (
            self.args.box_mode == "axis"
            and self.drag_start is not None
            and self.drag_current is not None
        ):
            box = normalized_box(
                (*self.drag_start, *self.drag_current), width, height
            )
            if box[2] > box[0] and box[3] > box[1]:
                return annotation_corners(box, width, height)
        return None

    def display_frame(self) -> Optional[np.ndarray]:
        frame = (
            self.frozen_frame.copy()
            if self.frozen_frame is not None
            else self.live_frame()
        )
        if frame is None:
            return None
        display = draw_boxes(frame, self.boxes, self.active_box())
        if (
            self.args.box_mode == "rotated"
            and self.drag_start is not None
            and self.drag_current is not None
        ):
            cv2.line(
                display, self.drag_start, self.drag_current, YELLOW, 2
            )
        mode = "FROZEN" if self.frozen_frame is not None else "LIVE"
        cv2.rectangle(display, (0, 0), (display.shape[1], 77), (0, 0, 0), -1)
        cv2.putText(
            display,
            "{} {} | left-drag:start  right-click:cancel/remove".format(
                mode, self.args.box_mode.upper()
            ),
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.57, WHITE, 1,
        )
        cv2.putText(
            display,
            "S:save  N:negative  U:undo box  C:clear  D:undo saved  Q:quit",
            (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
        )
        cv2.putText(
            display, self.status, (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.status_color, 1,
        )
        return display

    def save_positive(self) -> None:
        if self.pending_axis is not None or self.drag_start is not None:
            self.set_status(
                "Finish the rotated box or right-click to cancel it", RED
            )
            return
        if self.frozen_frame is None or not self.boxes:
            self.set_status("Draw at least one box before saving", RED)
            return
        try:
            path, count = self.writer.save(self.frozen_frame, self.boxes)
        except (OSError, ValueError) as error:
            self.set_status("Save failed: {}".format(error), RED)
            return
        print("Saved positive: {} ({} boxes)".format(path, count), flush=True)
        self.resume()

    def save_negative(self) -> None:
        frame = (
            self.frozen_frame.copy()
            if self.frozen_frame is not None
            else self.live_frame()
        )
        if frame is None:
            self.set_status("No camera frame to save", RED)
            return
        try:
            path, _ = self.writer.save(frame, [], negative=True)
        except (OSError, ValueError) as error:
            self.set_status("Save failed: {}".format(error), RED)
            return
        print("Saved negative: {}".format(path), flush=True)
        self.resume()

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key == ord(" "):
            self.freeze() if self.frozen_frame is None else self.resume()
        elif key in (ord("s"), ord("S"), 13):
            self.save_positive()
        elif key in (ord("n"), ord("N")):
            self.save_negative()
        elif key in (ord("u"), ord("U"), 8):
            if self.pending_axis is not None or self.drag_start is not None:
                self.pending_axis = None
                self.width_current = None
                self.drag_start = None
                self.drag_current = None
                self.set_status("Cancelled unfinished box", YELLOW)
            elif self.boxes:
                self.boxes.pop()
                self.set_status(
                    "Removed last box; {} remain".format(len(self.boxes)),
                    YELLOW,
                )
            else:
                self.set_status("No box to remove", RED)
        elif key in (ord("c"), ord("C")):
            self.boxes = []
            self.pending_axis = None
            self.width_current = None
            self.drag_start = None
            self.drag_current = None
            self.set_status("Cleared boxes", YELLOW)
        elif key in (ord("d"), ord("D")):
            path = self.writer.undo()
            if path is None:
                self.set_status("Nothing saved in this session to undo", RED)
            else:
                print("Removed saved sample: {}".format(path), flush=True)
                self.set_status("Removed last saved sample", YELLOW)
        return True

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        deadline = time.monotonic() + self.args.camera_timeout
        while not rospy.is_shutdown():
            display = self.display_frame()
            if display is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "timed out waiting for {}".format(self.args.topic)
                    )
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            cv2.imshow(WINDOW_NAME, display)
            if not self.handle_key(cv2.waitKey(15) & 0xFF):
                break
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Start/reuse D435 color, draw boxes with the mouse, and save YOLO labels."
        )
    )
    parser.add_argument("--topic", default="/d435/color/image_raw")
    parser.add_argument(
        "--dataset-name", default="manual_white_rectangle_dataset",
        help="folder under object_detective; ignored with --dataset-dir",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--class-name", default="white_rectangle")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--format", dest="label_format", choices=("obb", "detect"),
        default="obb", help="obb keeps the existing YOLO-OBB workflow",
    )
    parser.add_argument(
        "--box-mode", choices=("auto", "rotated", "axis"), default="auto",
        help="mouse interaction; auto uses rotated for OBB and axis for detect",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--camera-timeout", type=float, default=30.0)
    parser.add_argument(
        "--no-start-camera", action="store_true",
        help="subscribe only; do not start roscore or roslaunch",
    )
    parser.add_argument(
        "--camera-launch", default="d435_calibrated.launch",
        help="launch file in d435_eye_on_hand when the topic is absent",
    )
    args = parser.parse_args()
    if args.dataset_dir is None:
        name = args.dataset_name.strip()
        if not name or Path(name).name != name or name in (".", ".."):
            parser.error("--dataset-name must be one folder name without slashes")
        args.dataset_dir = script_dir / name
    if args.min_box_size < 2:
        parser.error("--min-box-size must be at least 2")
    if args.camera_timeout <= 0:
        parser.error("--camera-timeout must be positive")
    if args.box_mode == "auto":
        args.box_mode = "rotated" if args.label_format == "obb" else "axis"
    return args


def main() -> None:
    args = parse_args()
    managed: List[ManagedProcess] = []
    try:
        if not rosgraph.is_master_online():
            if args.no_start_camera:
                raise RuntimeError(
                    "ROS master is offline and --no-start-camera was set"
                )
            print("Starting roscore ...", flush=True)
            managed.append(ManagedProcess(["roscore"], "roscore"))
            if not wait_for_master(10.0):
                raise RuntimeError("roscore did not become ready")

        rospy.init_node("manual_dataset_annotator", anonymous=True)
        if not topic_is_published(args.topic):
            if args.no_start_camera:
                raise RuntimeError(
                    "{} is not published and --no-start-camera was set".format(
                        args.topic
                    )
                )
            print("Starting D435 color camera ...", flush=True)
            managed.append(ManagedProcess(
                ["roslaunch", "d435_eye_on_hand", args.camera_launch],
                "D435 camera",
            ))
        else:
            print("Reusing existing camera topic: {}".format(args.topic), flush=True)

        writer = DatasetWriter(
            args.dataset_dir, args.split, args.class_name, args.class_id,
            args.label_format, args.jpeg_quality,
        )
        print("Dataset: {}".format(writer.dataset_dir), flush=True)
        print("Split: {}  format: {}".format(args.split, args.label_format), flush=True)
        ManualAnnotator(args, writer).run()
    finally:
        if not rospy.core.is_shutdown():
            rospy.signal_shutdown("manual annotator exiting")
        for process in reversed(managed):
            print("Stopping {} ...".format(process.name), flush=True)
            process.stop()


if __name__ == "__main__":
    main()
