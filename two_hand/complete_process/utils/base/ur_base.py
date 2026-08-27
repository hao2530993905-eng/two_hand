from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

try:
    import rtde_control
    import rtde_receive
except ModuleNotFoundError:
    rtde_control = None
    rtde_receive = None

from .robotiq_gripper import GripperResult, RobotiqGripper


@dataclass
class MotionResult:
    success: bool
    message: str
    elapsed_s: float
    final_pose: Optional[List[float]] = None
    final_joints: Optional[List[float]] = None


@dataclass
class ReadResult:
    success: bool
    message: str
    value: Optional[List[float]] = None


def _as_6d_list(values: Iterable[float], name: str) -> List[float]:
    result = [float(v) for v in values]
    if len(result) != 6:
        raise ValueError(f"{name} must contain 6 values, got {len(result)}")
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return result


def _urscript_values(values: Sequence[float]) -> str:
    return ",".join(format(value, ".17g") for value in values)


def _max_abs_error(actual: Sequence[float], target: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(actual, target))


def _tcp_error(actual: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    position_error = math.sqrt(sum((actual[i] - target[i]) ** 2 for i in range(3)))
    rotation_error = math.sqrt(sum((actual[i] - target[i]) ** 2 for i in range(3, 6)))
    return position_error, rotation_error


class UR_BASE:
    """Minimal UR3 control wrapper extracted from the old UR_BASE.

    Only RTDE and optional Robotiq gripper control are kept here. Camera, ATI
    force sensor, Excel, and data collection dependencies are deliberately not
    imported.
    """

    def __init__(
        self,
        host: str = "192.168.1.5",
        gripper_port: Optional[int] = None,
        connect_control: bool = True,
        secondary_port: int = 30002,
    ) -> None:
        self.host = host
        self.gripper_port = gripper_port
        self.connect_control = connect_control
        self.secondary_port = int(secondary_port)
        self.rtde_c: Optional[Any] = None
        self.rtde_r: Optional[Any] = None
        self.gripper: Optional[RobotiqGripper] = None
        self.control_backend: Optional[str] = None
        self._active_script_motion: Optional[str] = None
        self.connected = False

    def __enter__(self) -> "UR_BASE":
        result = self.connect()
        if not result.success:
            raise RuntimeError(result.message)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.disconnect()

    def connect(self, timeout_s: float = 5.0) -> MotionResult:
        start = time.monotonic()
        if rtde_control is None or rtde_receive is None:
            return MotionResult(False, "rtde_control/rtde_receive is not installed", 0.0)

        try:
            control_error: Optional[str] = None
            if self.connect_control:
                flags = (
                    rtde_control.RTDEControlInterface.FLAGS_DEFAULT
                    | rtde_control.RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS
                )
                try:
                    self.rtde_c = rtde_control.RTDEControlInterface(self.host, flags=flags)
                    self.control_backend = "rtde"
                except Exception as exc:
                    self.rtde_c = None
                    control_error = str(exc)
                    if "RTDE input registers are already in use" not in control_error:
                        raise RuntimeError(f"RTDE control connection failed: {control_error}") from exc
                    try:
                        self._check_urscript_control(timeout_s)
                    except Exception as fallback_exc:
                        raise RuntimeError(
                            f"RTDE control connection failed: {control_error}; {fallback_exc}"
                        ) from fallback_exc
                    self.control_backend = "urscript"
            try:
                self.rtde_r = rtde_receive.RTDEReceiveInterface(
                    self.host,
                    use_upper_range_registers=True,
                )
            except Exception as exc:
                raise RuntimeError(f"RTDE receive connection failed: {exc}") from exc

            if self.gripper_port is not None:
                self.gripper = RobotiqGripper()
                grip = self.gripper.connect(self.host, self.gripper_port, timeout_s=timeout_s)
                if not grip.success:
                    raise RuntimeError(f"Robot connected but gripper failed: {grip.message}")

            self.connected = True
            if not self.connect_control:
                self.control_backend = "receive_only"
            if control_error is not None:
                message = f"connected using URScript fallback; RTDE control unavailable: {control_error}"
            else:
                message = f"connected using {self.control_backend}"
            return MotionResult(True, message, time.monotonic() - start)
        except Exception as exc:
            self.disconnect()
            return MotionResult(False, f"connect failed: {exc}", time.monotonic() - start)

    def _check_urscript_control(self, timeout_s: float) -> None:
        timeout = _positive_float(timeout_s, "timeout_s")
        try:
            with socket.create_connection((self.host, self.secondary_port), timeout=timeout):
                pass
        except OSError as exc:
            raise RuntimeError(f"URScript fallback connection failed: {exc}") from exc

    def _send_urscript(self, statement: str, timeout_s: float = 2.0) -> None:
        timeout = _positive_float(timeout_s, "timeout_s")
        program = f"def workpiece_command():\n  {statement}\nend\n"
        try:
            with socket.create_connection((self.host, self.secondary_port), timeout=timeout) as client:
                client.sendall(program.encode("ascii"))
                client.shutdown(socket.SHUT_WR)
                header = self._recv_exact(client, 4)
                packet_size = struct.unpack(">I", header)[0]
                if packet_size < 5 or packet_size > 10_000_000:
                    raise RuntimeError(f"invalid Secondary Interface packet size: {packet_size}")
                self._recv_exact(client, packet_size - 4)
        except OSError as exc:
            raise RuntimeError(f"URScript send failed: {exc}") from exc

    @staticmethod
    def _recv_exact(client: socket.socket, size: int) -> bytes:
        chunks: List[bytes] = []
        remaining = size
        while remaining:
            chunk = client.recv(remaining)
            if not chunk:
                raise ConnectionError("Secondary Interface closed before a complete state packet was received")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def disconnect(self) -> MotionResult:
        start = time.monotonic()
        errors: List[str] = []

        if self.control_backend == "urscript" and self._active_script_motion is not None:
            if self._active_script_motion == "joint":
                stop = self.stop_j()
            else:
                stop = self.stop_l()
            if not stop.success:
                errors.append(stop.message)

        if self.rtde_c is not None:
            for method_name in ("servoStop", "stopScript", "disconnect"):
                try:
                    getattr(self.rtde_c, method_name)()
                except Exception as exc:
                    errors.append(f"{method_name}: {exc}")

        if self.rtde_r is not None and hasattr(self.rtde_r, "disconnect"):
            try:
                self.rtde_r.disconnect()
            except Exception as exc:
                errors.append(f"receive disconnect: {exc}")

        if self.gripper is not None:
            result = self.gripper.disconnect()
            if not result.success:
                errors.append(result.message)

        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None
        self.control_backend = None
        self._active_script_motion = None
        self.connected = False
        if errors:
            return MotionResult(False, "; ".join(errors), time.monotonic() - start)
        return MotionResult(True, "disconnected", time.monotonic() - start)

    def _require_receive(self) -> Optional[str]:
        if not self.connected or self.rtde_r is None:
            return "UR_BASE is not connected. Call connect() first."
        return None

    def _require_motion_control(self) -> Optional[str]:
        receive_error = self._require_receive()
        if receive_error:
            return receive_error
        if self.control_backend not in ("rtde", "urscript"):
            return "Motion control is not connected. Use connect_control=True."
        return None

    def _require_rtde_control(self) -> Optional[str]:
        receive_error = self._require_receive()
        if receive_error:
            return receive_error
        if self.control_backend != "rtde" or self.rtde_c is None:
            return "This operation requires RTDE control; the URScript fallback only supports move_j and move_l."
        return None

    def _urscript_motion_error(self) -> Optional[str]:
        try:
            robot_status = int(self.rtde_r.getRobotStatus())
        except Exception as exc:
            return f"Cannot verify robot program state before URScript motion: {exc}"
        if robot_status & 0b0010:
            return "Cannot send URScript motion while another robot program is running."
        return None

    def _wait_for_urscript_program_stop(self, timeout_s: float = 2.0, poll_s: float = 0.02) -> Optional[str]:
        deadline = time.monotonic() + _positive_float(timeout_s, "timeout_s")
        while time.monotonic() < deadline:
            try:
                if not (int(self.rtde_r.getRobotStatus()) & 0b0010):
                    return None
            except Exception as exc:
                return f"Cannot verify URScript motion completion: {exc}"
            time.sleep(poll_s)
        return "URScript motion reached its target but the robot program did not stop"

    def get_tcp_pose(self) -> ReadResult:
        error = self._require_receive()
        if error:
            return ReadResult(False, error)
        try:
            return ReadResult(True, "ok", list(self.rtde_r.getActualTCPPose()))
        except Exception as exc:
            return ReadResult(False, f"get_tcp_pose failed: {exc}")

    def get_tcp_speed(self) -> ReadResult:
        error = self._require_receive()
        if error:
            return ReadResult(False, error)
        try:
            return ReadResult(True, "ok", list(self.rtde_r.getActualTCPSpeed()))
        except Exception as exc:
            return ReadResult(False, f"get_tcp_speed failed: {exc}")

    def get_joint_positions(self) -> ReadResult:
        error = self._require_receive()
        if error:
            return ReadResult(False, error)
        try:
            return ReadResult(True, "ok", list(self.rtde_r.getActualQ()))
        except Exception as exc:
            return ReadResult(False, f"get_joint_positions failed: {exc}")

    def get_join_positions(self) -> ReadResult:
        """Backward-compatible alias for the common typo in notes."""
        return self.get_joint_positions()

    def get_tcp_pose_or_raise(self) -> List[float]:
        result = self.get_tcp_pose()
        if not result.success or result.value is None:
            raise RuntimeError(result.message)
        return result.value

    def get_joint_positions_or_raise(self) -> List[float]:
        result = self.get_joint_positions()
        if not result.success or result.value is None:
            raise RuntimeError(result.message)
        return result.value

    def wait_for_joint_positions(
        self,
        target: Iterable[float],
        timeout_s: float,
        tolerance_rad: float = 0.001,
        poll_s: float = 0.02,
    ) -> MotionResult:
        start = time.monotonic()
        target_q = _as_6d_list(target, "target")
        final_q: Optional[List[float]] = None

        while time.monotonic() - start < timeout_s:
            read = self.get_joint_positions()
            if not read.success or read.value is None:
                return MotionResult(False, read.message, time.monotonic() - start, final_joints=final_q)
            final_q = read.value
            if _max_abs_error(final_q, target_q) <= tolerance_rad:
                return MotionResult(True, "joint target reached", time.monotonic() - start, final_joints=final_q)
            time.sleep(poll_s)

        return MotionResult(False, "joint target timeout", time.monotonic() - start, final_joints=final_q)

    def wait_for_tcp_pose(
        self,
        target: Iterable[float],
        timeout_s: float,
        position_tolerance_m: float = 0.001,
        rotation_tolerance_rad: float = 0.01,
        poll_s: float = 0.02,
    ) -> MotionResult:
        start = time.monotonic()
        target_pose = _as_6d_list(target, "target")
        final_pose: Optional[List[float]] = None

        while time.monotonic() - start < timeout_s:
            read = self.get_tcp_pose()
            if not read.success or read.value is None:
                return MotionResult(False, read.message, time.monotonic() - start, final_pose=final_pose)
            final_pose = read.value
            pos_err, rot_err = _tcp_error(final_pose, target_pose)
            if pos_err <= position_tolerance_m and rot_err <= rotation_tolerance_rad:
                return MotionResult(True, "tcp target reached", time.monotonic() - start, final_pose=final_pose)
            time.sleep(poll_s)

        return MotionResult(False, "tcp target timeout", time.monotonic() - start, final_pose=final_pose)

    def move_j(
        self,
        joints: Iterable[float],
        speed: float = 0.5,
        acceleration: float = 0.5,
        timeout_s: float = 10.0,
        tolerance_rad: float = 0.001,
    ) -> MotionResult:
        start = time.monotonic()
        error = self._require_motion_control()
        if error:
            return MotionResult(False, error, 0.0)
        target = _as_6d_list(joints, "joints")
        speed_value = _positive_float(speed, "speed")
        acceleration_value = _positive_float(acceleration, "acceleration")

        try:
            if self.control_backend == "rtde":
                command_ok = self.rtde_c.moveJ(target, speed_value, acceleration_value, True)
            else:
                program_error = self._urscript_motion_error()
                if program_error:
                    return MotionResult(False, program_error, time.monotonic() - start)
                statement = (
                    f"movej([{_urscript_values(target)}], "
                    f"a={format(acceleration_value, '.17g')}, v={format(speed_value, '.17g')})"
                )
                self._active_script_motion = "joint"
                self._send_urscript(statement)
                command_ok = True
        except Exception as exc:
            if self.control_backend == "urscript" and self._active_script_motion == "joint":
                stop = self.stop_j()
                return MotionResult(
                    False,
                    f"moveJ command failed: {exc}; emergency stop: {stop.message}",
                    time.monotonic() - start,
                )
            return MotionResult(False, f"moveJ command failed: {exc}", time.monotonic() - start)

        if command_ok is False:
            return MotionResult(False, "moveJ command rejected", time.monotonic() - start)

        result = self.wait_for_joint_positions(target, timeout_s, tolerance_rad)
        if not result.success:
            stop = self.stop_j()
            if not stop.success:
                result.message = f"{result.message}; {stop.message}"
        elif self.control_backend == "urscript":
            completion_error = self._wait_for_urscript_program_stop()
            if completion_error:
                result.success = False
                result.message = completion_error
                stop = self.stop_j()
                if not stop.success:
                    result.message = f"{result.message}; {stop.message}"
            else:
                self._active_script_motion = None
        result.elapsed_s = time.monotonic() - start
        return result

    def move_l(
        self,
        pose: Iterable[float],
        speed: float = 0.03,
        acceleration: float = 0.1,
        timeout_s: float = 10.0,
        position_tolerance_m: float = 0.001,
        rotation_tolerance_rad: float = 0.01,
    ) -> MotionResult:
        start = time.monotonic()
        error = self._require_motion_control()
        if error:
            return MotionResult(False, error, 0.0)
        target = _as_6d_list(pose, "pose")
        speed_value = _positive_float(speed, "speed")
        acceleration_value = _positive_float(acceleration, "acceleration")

        try:
            if self.control_backend == "rtde":
                command_ok = self.rtde_c.moveL(target, speed_value, acceleration_value, True)
            else:
                program_error = self._urscript_motion_error()
                if program_error:
                    return MotionResult(False, program_error, time.monotonic() - start)
                statement = (
                    f"movel(p[{_urscript_values(target)}], "
                    f"a={format(acceleration_value, '.17g')}, v={format(speed_value, '.17g')})"
                )
                self._active_script_motion = "linear"
                self._send_urscript(statement)
                command_ok = True
        except Exception as exc:
            if self.control_backend == "urscript" and self._active_script_motion == "linear":
                stop = self.stop_l()
                return MotionResult(
                    False,
                    f"moveL command failed: {exc}; emergency stop: {stop.message}",
                    time.monotonic() - start,
                )
            return MotionResult(False, f"moveL command failed: {exc}", time.monotonic() - start)

        if command_ok is False:
            return MotionResult(False, "moveL command rejected", time.monotonic() - start)

        result = self.wait_for_tcp_pose(target, timeout_s, position_tolerance_m, rotation_tolerance_rad)
        if not result.success:
            stop = self.stop_l()
            if not stop.success:
                result.message = f"{result.message}; {stop.message}"
        elif self.control_backend == "urscript":
            completion_error = self._wait_for_urscript_program_stop()
            if completion_error:
                result.success = False
                result.message = completion_error
                stop = self.stop_l()
                if not stop.success:
                    result.message = f"{result.message}; {stop.message}"
            else:
                self._active_script_motion = None
        result.elapsed_s = time.monotonic() - start
        return result

    def move_j_ik(
        self,
        pose: Iterable[float],
        speed: float = 0.5,
        acceleration: float = 0.5,
        timeout_s: float = 10.0,
        position_tolerance_m: float = 0.001,
        rotation_tolerance_rad: float = 0.01,
    ) -> MotionResult:
        """MoveJ to a TCP pose, using inverse kinematics to obtain joint targets."""
        start = time.monotonic()
        error = self._require_motion_control()
        if error:
            return MotionResult(False, error, 0.0)
        target = _as_6d_list(pose, "pose")
        speed_value = _positive_float(speed, "speed")
        acceleration_value = _positive_float(acceleration, "acceleration")

        try:
            if self.control_backend == "rtde":
                # getInverseKinematicsHasSolution() is unreliable on the
                # project's UR3 CB3 + ur_rtde 1.6.0 combination: it returns
                # False even for getActualTCPPose(). Fetch the actual solution
                # instead and validate the returned joint vector.
                qnear = list(self.rtde_r.getActualQ())
                joint_target = list(
                    self.rtde_c.getInverseKinematics(
                        target,
                        qnear,
                        1e-5,
                        1e-5,
                    )
                )
                if (
                    len(joint_target) != 6
                    or not all(math.isfinite(value) for value in joint_target)
                ):
                    tcp_offset = []
                    try:
                        tcp_offset = list(self.rtde_c.getTCPOffset())
                    except Exception:
                        pass
                    return MotionResult(
                        False,
                        "controller returned no inverse-kinematics solution "
                        f"for active-TCP pose {target}; active TCP offset={tcp_offset}; "
                        "verify that the pose was recorded in UR Base with the same TCP",
                        time.monotonic() - start,
                    )
                command_ok = self.rtde_c.moveJ(
                    joint_target,
                    speed_value,
                    acceleration_value,
                    True,
                )
            else:
                program_error = self._urscript_motion_error()
                if program_error:
                    return MotionResult(False, program_error, time.monotonic() - start)
                statement = (
                    f"movej(get_inverse_kin(p[{_urscript_values(target)}]), "
                    f"a={format(acceleration_value, '.17g')}, v={format(speed_value, '.17g')})"
                )
                self._active_script_motion = "joint"
                self._send_urscript(statement)
                command_ok = True
        except Exception as exc:
            if self._active_script_motion == "joint":
                stop = self.stop_j()
                return MotionResult(
                    False,
                    f"moveJ IK command failed: {exc}; emergency stop: {stop.message}",
                    time.monotonic() - start,
                )
            return MotionResult(False, f"moveJ IK command failed: {exc}", time.monotonic() - start)

        if command_ok is False:
            return MotionResult(False, "moveJ IK command rejected", time.monotonic() - start)

        result = self.wait_for_tcp_pose(
            target,
            timeout_s,
            position_tolerance_m,
            rotation_tolerance_rad,
        )
        if not result.success:
            stop = self.stop_j()
            if not stop.success:
                result.message = f"{result.message}; {stop.message}"
        elif self.control_backend == "urscript":
            completion_error = self._wait_for_urscript_program_stop()
            if completion_error:
                result.success = False
                result.message = completion_error
                stop = self.stop_j()
                if not stop.success:
                    result.message = f"{result.message}; {stop.message}"
            else:
                self._active_script_motion = None
        result.elapsed_s = time.monotonic() - start
        return result

    def servo_l(
        self,
        pose: Iterable[float],
        dt_s: float = 0.02,
        speed: float = 0.0,
        acceleration: float = 0.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
    ) -> MotionResult:
        start = time.monotonic()
        error = self._require_rtde_control()
        if error:
            return MotionResult(False, error, 0.0)
        target = _as_6d_list(pose, "pose")

        try:
            command_ok = self.rtde_c.servoL(target, speed, acceleration, dt_s, lookahead_time, gain)
        except Exception as exc:
            return MotionResult(False, f"servoL command failed: {exc}", time.monotonic() - start)

        if command_ok is False:
            return MotionResult(False, "servoL command rejected", time.monotonic() - start)
        return MotionResult(True, "servoL command sent", time.monotonic() - start)

    def follow_servo_path(
        self,
        path: Iterable[Iterable[float]],
        dt_s: float = 0.02,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        path_timeout_s: Optional[float] = None,
        final_timeout_s: float = 2.0,
        position_tolerance_m: float = 0.002,
        rotation_tolerance_rad: float = 0.02,
    ) -> MotionResult:
        start = time.monotonic()
        error = self._require_rtde_control()
        if error:
            return MotionResult(False, error, 0.0)

        points = [_as_6d_list(pose, "path pose") for pose in path]
        if not points:
            return MotionResult(False, "empty servo path", 0.0)

        if path_timeout_s is None:
            path_timeout_s = max(1.0, len(points) * dt_s * 2.0)

        try:
            for pose in points:
                if time.monotonic() - start > path_timeout_s:
                    self.stop_servo()
                    return MotionResult(False, "servo path timeout", time.monotonic() - start, final_pose=self.get_tcp_pose().value)

                period_start = self.rtde_c.initPeriod()
                command = self.servo_l(pose, dt_s=dt_s, lookahead_time=lookahead_time, gain=gain)
                if not command.success:
                    self.stop_servo()
                    return MotionResult(False, command.message, time.monotonic() - start, final_pose=self.get_tcp_pose().value)
                self.rtde_c.waitPeriod(period_start)
        finally:
            self.stop_servo()

        result = self.wait_for_tcp_pose(points[-1], final_timeout_s, position_tolerance_m, rotation_tolerance_rad)
        result.elapsed_s = time.monotonic() - start
        return result

    def stop_servo(self) -> MotionResult:
        start = time.monotonic()
        if self.rtde_c is None:
            return MotionResult(True, "no control connection", 0.0)
        try:
            self.rtde_c.servoStop()
            return MotionResult(True, "servo stopped", time.monotonic() - start)
        except Exception as exc:
            return MotionResult(False, f"servoStop failed: {exc}", time.monotonic() - start)

    def stop_l(self, acceleration: float = 0.5) -> MotionResult:
        start = time.monotonic()
        if self.control_backend not in ("rtde", "urscript"):
            return MotionResult(True, "no control connection", 0.0)
        try:
            acceleration_value = _positive_float(acceleration, "acceleration")
            if self.control_backend == "rtde":
                self.rtde_c.stopL(acceleration_value)
            else:
                self._send_urscript(f"stopl({format(acceleration_value, '.17g')})")
                self._active_script_motion = None
            return MotionResult(True, "linear motion stopped", time.monotonic() - start)
        except Exception as exc:
            if self.control_backend == "urscript":
                return MotionResult(False, f"stopL failed: {exc}", time.monotonic() - start)
            servo = self.stop_servo()
            return MotionResult(servo.success, f"stopL failed: {exc}; {servo.message}", time.monotonic() - start)

    def stop_j(self, acceleration: float = 1.5) -> MotionResult:
        start = time.monotonic()
        if self.control_backend not in ("rtde", "urscript"):
            return MotionResult(True, "no control connection", 0.0)
        try:
            acceleration_value = _positive_float(acceleration, "acceleration")
            if self.control_backend == "rtde":
                self.rtde_c.stopJ(acceleration_value)
            else:
                self._send_urscript(f"stopj({format(acceleration_value, '.17g')})")
                self._active_script_motion = None
            return MotionResult(True, "joint motion stopped", time.monotonic() - start)
        except Exception as exc:
            if self.control_backend == "urscript":
                return MotionResult(False, f"stopJ failed: {exc}", time.monotonic() - start)
            servo = self.stop_servo()
            return MotionResult(servo.success, f"stopJ failed: {exc}; {servo.message}", time.monotonic() - start)

    def open_gripper(self, position: int = 0, speed: int = 255, force: int = 255, timeout_s: float = 5.0) -> GripperResult:
        if self.gripper is None:
            return GripperResult(False, "gripper is not configured", None)
        return self.gripper.move_and_wait_for_pos(position, speed, force, timeout_s)

    def close_gripper(self, position: int = 255, speed: int = 255, force: int = 255, timeout_s: float = 5.0) -> GripperResult:
        if self.gripper is None:
            return GripperResult(False, "gripper is not configured", None)
        return self.gripper.move_and_wait_for_pos(position, speed, force, timeout_s)


URBase = UR_BASE
