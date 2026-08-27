from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GripperResult:
    success: bool
    message: str
    position: Optional[int] = None


class RobotiqGripper:
    """Small socket client for Robotiq grippers on UR robots."""

    ENCODING = "UTF-8"
    ACK = b"ack"

    POS = "POS"
    PRE = "PRE"
    OBJ = "OBJ"
    SPE = "SPE"
    FOR = "FOR"
    GTO = "GTO"
    ACT = "ACT"
    ATR = "ATR"

    def __init__(self) -> None:
        self.socket: Optional[socket.socket] = None
        self.command_lock = threading.Lock()
        self.host: Optional[str] = None
        self.port: Optional[int] = None

    def connect(self, hostname: str, port: int = 63352, timeout_s: float = 5.0) -> GripperResult:
        try:
            self.host = hostname
            self.port = port
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout_s)
            self.socket.connect((hostname, port))
            reset = self.reset()
            if not reset.success:
                return reset
            return self.activate(timeout_s=timeout_s)
        except Exception as exc:
            self.disconnect()
            return GripperResult(False, f"gripper connect failed: {exc}")

    def disconnect(self) -> GripperResult:
        if self.socket is None:
            return GripperResult(True, "already disconnected")
        try:
            self.socket.close()
            return GripperResult(True, "disconnected")
        except Exception as exc:
            return GripperResult(False, f"gripper disconnect failed: {exc}")
        finally:
            self.socket = None

    def _require_socket(self) -> Optional[str]:
        if self.socket is None:
            return "gripper is not connected"
        return None

    def _set_vars(self, values: dict[str, int]) -> GripperResult:
        error = self._require_socket()
        if error:
            return GripperResult(False, error)
        cmd = "SET" + "".join(f" {name} {int(value)}" for name, value in values.items()) + "\n"
        try:
            with self.command_lock:
                self.socket.sendall(cmd.encode(self.ENCODING))
                data = self.socket.recv(1024)
            if data == self.ACK:
                return GripperResult(True, "ack")
            return GripperResult(False, f"unexpected gripper response: {data!r}")
        except Exception as exc:
            return GripperResult(False, f"gripper SET failed: {exc}")

    def _get_var(self, name: str) -> tuple[bool, Optional[int], str]:
        error = self._require_socket()
        if error:
            return False, None, error
        cmd = f"GET {name}\n"
        try:
            with self.command_lock:
                self.socket.sendall(cmd.encode(self.ENCODING))
                data = self.socket.recv(1024)
            decoded = data.decode(self.ENCODING).strip()
            parts = decoded.split()
            if len(parts) != 2 or parts[0] != name:
                return False, None, f"unexpected gripper response: {decoded}"
            return True, int(parts[1]), "ok"
        except Exception as exc:
            return False, None, f"gripper GET {name} failed: {exc}"

    def get_force_setting(self) -> tuple[bool, Optional[int], str]:
        """Read the configured force register (0-255), not a measured force in newtons."""
        return self._get_var(self.FOR)

    def get_position(self) -> tuple[bool, Optional[int], str]:
        return self._get_var(self.POS)

    def get_object_status(self) -> tuple[bool, Optional[int], str]:
        return self._get_var(self.OBJ)

    def reset(self) -> GripperResult:
        return self._set_vars({self.ACT: 0, self.ATR: 0})

    def activate(self, timeout_s: float = 5.0) -> GripperResult:
        result = self._set_vars({self.ACT: 1})
        if not result.success:
            return result
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, value, message = self._get_var(self.ACT)
            if not ok:
                return GripperResult(False, message)
            if value == 1:
                return GripperResult(True, "activated")
            time.sleep(0.05)
        return GripperResult(False, "gripper activation timeout")

    def move(self, position: int, speed: int = 255, force: int = 255) -> GripperResult:
        clipped_position = max(0, min(255, int(position)))
        clipped_speed = max(0, min(255, int(speed)))
        clipped_force = max(0, min(255, int(force)))
        result = self._set_vars({
            self.POS: clipped_position,
            self.SPE: clipped_speed,
            self.FOR: clipped_force,
            self.GTO: 1,
        })
        if not result.success:
            return result
        return GripperResult(True, "move command sent", clipped_position)

    def move_and_wait_for_pos(
        self,
        position: int,
        speed: int = 255,
        force: int = 255,
        timeout_s: float = 5.0,
    ) -> GripperResult:
        command = self.move(position, speed, force)
        if not command.success:
            return command

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok_pre, pre, msg_pre = self._get_var(self.PRE)
            if not ok_pre:
                return GripperResult(False, msg_pre)
            ok_obj, obj, msg_obj = self._get_var(self.OBJ)
            if not ok_obj:
                return GripperResult(False, msg_obj)
            ok_pos, actual_pos, msg_pos = self._get_var(self.POS)
            if not ok_pos:
                return GripperResult(False, msg_pos)

            if pre == command.position and obj in (1, 2, 3):
                return GripperResult(True, f"gripper reached state OBJ={obj}", actual_pos)
            time.sleep(0.02)

        ok_pos, actual_pos, _ = self._get_var(self.POS)
        return GripperResult(False, "gripper move timeout", actual_pos if ok_pos else None)

