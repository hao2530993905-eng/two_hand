from __future__ import annotations

import unittest
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import _path_setup  # noqa: F401
import complete_process.utils.base.ur_base as ur_base
from complete_process.utils.base.ur_base import MotionResult, UR_BASE


class FailingRTDEControl:
    FLAGS_DEFAULT = 1
    FLAG_UPPER_RANGE_REGISTERS = 8

    def __init__(self, host: str, flags: int) -> None:
        raise RuntimeError("One of the RTDE input registers are already in use")


class URBaseFallbackTests(unittest.TestCase):
    def test_connect_falls_back_to_urscript(self) -> None:
        receive = MagicMock()
        rtde_receive = SimpleNamespace(RTDEReceiveInterface=MagicMock(return_value=receive))
        rtde_control = SimpleNamespace(RTDEControlInterface=FailingRTDEControl)
        connection = MagicMock()

        with patch.object(ur_base, "rtde_control", rtde_control):
            with patch.object(ur_base, "rtde_receive", rtde_receive):
                with patch.object(
                    ur_base.socket,
                    "create_connection",
                    return_value=connection,
                ) as create_connection:
                    robot = UR_BASE("192.0.2.1")
                    result = robot.connect(timeout_s=1.0)

        self.assertTrue(result.success)
        self.assertEqual(robot.control_backend, "urscript")
        self.assertIn("RTDE control unavailable", result.message)
        create_connection.assert_called_once_with(("192.0.2.1", 30002), timeout=1.0)
        rtde_receive.RTDEReceiveInterface.assert_called_once_with(
            "192.0.2.1",
            use_upper_range_registers=True,
        )

    def test_move_j_sends_urscript_and_waits_for_target(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "urscript"
        robot.rtde_r = MagicMock()
        robot.rtde_r.getRobotStatus.return_value = 1
        robot._send_urscript = MagicMock()
        robot.wait_for_joint_positions = MagicMock(
            return_value=MotionResult(True, "joint target reached", 0.1, final_joints=[0.0] * 6)
        )

        result = robot.move_j([1, 2, 3, 4, 5, 6], speed=0.1, acceleration=0.2)

        self.assertTrue(result.success)
        robot._send_urscript.assert_called_once_with("movej([1,2,3,4,5,6], a=0.20000000000000001, v=0.10000000000000001)")
        robot.wait_for_joint_positions.assert_called_once_with([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 10.0, 0.001)

    def test_move_l_formats_pose_and_waits_for_target(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "urscript"
        robot.rtde_r = MagicMock()
        robot.rtde_r.getRobotStatus.return_value = 1
        robot._send_urscript = MagicMock()
        robot.wait_for_tcp_pose = MagicMock(
            return_value=MotionResult(True, "tcp target reached", 0.1, final_pose=[0.0] * 6)
        )

        result = robot.move_l([0.1, 0.2, 0.3, 0, 0, 0], speed=0.01, acceleration=0.03)

        self.assertTrue(result.success)
        statement = robot._send_urscript.call_args.args[0]
        self.assertTrue(statement.startswith("movel(p["))
        self.assertIn("a=0.029999999999999999", statement)
        self.assertIn("v=0.01", statement)
        robot.wait_for_tcp_pose.assert_called_once()

    def test_move_j_ik_checks_solution_and_waits_for_tcp_pose(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "rtde"
        robot.rtde_r = MagicMock()
        robot.rtde_c = MagicMock()
        robot.rtde_c.getInverseKinematicsHasSolution.return_value = True
        robot.rtde_c.moveJ_IK.return_value = True
        robot.wait_for_tcp_pose = MagicMock(
            return_value=MotionResult(True, "tcp target reached", 0.1, final_pose=[0.0] * 6)
        )
        pose = [0.1, 0.2, 0.3, 3.14, 0.0, 0.0]

        result = robot.move_j_ik(pose, speed=0.2, acceleration=0.4)

        self.assertTrue(result.success)
        robot.rtde_c.getInverseKinematicsHasSolution.assert_called_once_with(pose)
        robot.rtde_c.moveJ_IK.assert_called_once_with(pose, 0.2, 0.4, True)
        robot.wait_for_tcp_pose.assert_called_once()

    def test_move_j_ik_rejects_pose_without_solution(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "rtde"
        robot.rtde_r = MagicMock()
        robot.rtde_c = MagicMock()
        robot.rtde_c.getInverseKinematicsHasSolution.return_value = False

        result = robot.move_j_ik([0.1, 0.2, 0.3, 3.14, 0.0, 0.0])

        self.assertFalse(result.success)
        self.assertIn("no inverse-kinematics solution", result.message)
        robot.rtde_c.moveJ_IK.assert_not_called()

    def test_servo_l_reports_that_fallback_is_not_supported(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "urscript"
        robot.rtde_r = MagicMock()

        result = robot.servo_l([0.0] * 6)

        self.assertFalse(result.success)
        self.assertIn("requires RTDE control", result.message)

    def test_urscript_motion_does_not_replace_a_running_program(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "urscript"
        robot.rtde_r = MagicMock()
        robot.rtde_r.getRobotStatus.return_value = 0b0011
        robot._send_urscript = MagicMock()

        result = robot.move_j([0.0] * 6)

        self.assertFalse(result.success)
        self.assertIn("another robot program is running", result.message)
        robot._send_urscript.assert_not_called()

    def test_urscript_stop_failure_is_reported(self) -> None:
        robot = UR_BASE()
        robot.connected = True
        robot.control_backend = "urscript"
        robot.rtde_r = MagicMock()
        robot._send_urscript = MagicMock(side_effect=RuntimeError("network error"))

        result = robot.stop_j()

        self.assertFalse(result.success)
        self.assertIn("network error", result.message)

    def test_send_urscript_reads_one_state_packet_before_closing(self) -> None:
        robot = UR_BASE("192.0.2.1")
        client = MagicMock()
        client.recv.side_effect = [struct.pack(">I", 5), b"\x10"]
        connection = MagicMock()
        connection.__enter__.return_value = client

        with patch.object(ur_base.socket, "create_connection", return_value=connection):
            robot._send_urscript('textmsg("delivery test")', timeout_s=1.0)

        client.sendall.assert_called_once_with(
            b'def workpiece_command():\n  textmsg("delivery test")\nend\n'
        )
        client.shutdown.assert_called_once_with(ur_base.socket.SHUT_WR)


if __name__ == "__main__":
    unittest.main()
