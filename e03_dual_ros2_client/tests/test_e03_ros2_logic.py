"""Exercise ROS control decisions with message/action doubles, without hardware.

These tests do not substitute for a ROS2 graph/driver integration test.
"""

import collections
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
import threading
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_e03_client import configured_robot


@pytest.fixture
def ros_logic(monkeypatch):
    def module(name, **values):
        result = types.ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)

    def pose():
        return NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=0.))

    def goal():
        return NS(request=NS(start_state=NS()), planning_options=NS(
            planning_scene_diff=NS(robot_state=NS())))

    class Time:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = nanoseconds

    module("rclpy", ok=lambda: True)
    module("rclpy.action", ActionClient=object)
    module("rclpy.duration", Duration=lambda **kwargs: NS(**kwargs))
    module("rclpy.node", Node=type("Node", (), {}))
    module("rclpy.qos", qos_profile_sensor_data=None)
    module("rclpy.time", Time=Time)
    module("action_msgs.msg", GoalStatus=NS(STATUS_SUCCEEDED=4))
    module("control_msgs.action", GripperCommand=NS(Goal=lambda: NS(command=NS())))
    module("cv_bridge", CvBridge=object)
    module("geometry_msgs.msg", Pose=pose)
    module("moveit_msgs.action", MoveGroup=NS(Goal=goal))
    module("moveit_msgs.msg", Constraints=lambda: NS(position_constraints=[], orientation_constraints=[]),
           MoveItErrorCodes=NS(SUCCESS=1),
           OrientationConstraint=lambda: NS(header=NS()),
           PositionConstraint=lambda: NS(header=NS(), constraint_region=NS()))
    module("sensor_msgs.msg", CameraInfo=object, Image=object, JointState=object)
    module("shape_msgs.msg", SolidPrimitive=type("Primitive", (), {"SPHERE": 2}))
    module("tf2_ros", Buffer=object, TransformListener=object)
    path = Path(__file__).resolve().parents[1] / "clients/e03_ros2_node.py"
    spec = importlib.util.spec_from_file_location("clients._ros_logic_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_dual_goal_requires_both_poses_in_one_constraint_set(ros_logic):
    config = configured_robot()
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    targets[:, 1, 3] = [0.2, -0.2]
    goal = ros_logic.move_goal(targets, config, plan_only=True)
    assert goal.request.group_name == "both_arms"
    assert goal.planning_options.plan_only is True
    assert len(goal.request.goal_constraints) == 1  # Alternatives here would be wrong.
    constraints = goal.request.goal_constraints[0]
    assert [x.link_name for x in constraints.position_constraints] == ["left_tcp", "right_tcp"]
    assert [x.constraint_region.primitive_poses[0].position.y
            for x in constraints.position_constraints] == [0.2, -0.2]
    assert len(constraints.orientation_constraints) == 2
    assert all(x.header.frame_id == "world" for x in constraints.position_constraints)


def test_capture_uses_two_tcp_poses_rectified_camera_frame_and_relative_time(ros_logic):
    node = ros_logic.E03Node.__new__(ros_logic.E03Node)
    node.config = configured_robot()
    node.config["cameras"] = {"external_cam": {"rectified": True}}
    node.config["arms"][0]["gripper"]["fixed_model_value"] = 0.2
    node.config["arms"][1]["gripper"]["fixed_model_value"] = 0.8
    node.stopping, node.origin_ns, node.last_stamp_ns, node.sequence = False, None, -1, 0
    node.capture_lock, node.frames = threading.Lock(), collections.deque(maxlen=10)
    node.last_sensor_error = ""
    joint_names = [name for arm in node.config["arms"] for name in arm["joint_names"]]
    node.joints = {name: (0.0, 2_000_000_000) for name in joint_names}
    header = NS(stamp=NS(sec=2, nanosec=0), frame_id="camera_optical")
    node.images = {"external_cam": NS(header=header, width=4, height=3)}
    rectification = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    node.intrinsics = {"external_cam": NS(width=4, height=3, header=header,
                         p=[100, 0, 2, 0, 0, 100, 1, 0, 0, 0, 1, 0], r=rectification.ravel())}
    node.bridge = NS(imgmsg_to_cv2=lambda *args, **kwargs: np.zeros((3, 4, 3), np.uint8))
    clock = NS(nanoseconds=2_000_000_000)
    node.get_clock = lambda: NS(now=lambda: clock)
    lookups = []

    def tf(child, when):
        lookups.append((child, when.nanoseconds))
        matrix = np.eye(4)
        matrix[1, 3] = {"left_tcp": 0.2, "right_tcp": -0.2, "camera_optical": 0.0}[child]
        return matrix, when.nanoseconds

    node._tf_matrix = tf
    node._capture()
    assert node.last_sensor_error == ""
    frame = node.frames[-1][2]
    assert frame["timestamp"] == 0
    np.testing.assert_allclose(frame["ee_pose"][:, 1, 3], [0.2, -0.2])
    np.testing.assert_allclose(frame["gripper"], [0.2, 0.8])
    np.testing.assert_allclose(frame["external_cam"]["data"]["wcT"][:3, :3], rectification.T)
    assert all(timestamp == 2_000_000_000 for _, timestamp in lookups)
    # Holding old feedback must not make it fresh again just by sampling a timer.
    clock.nanoseconds = 3_000_000_000
    node._capture()
    assert len(node.frames) == 1
    assert "stale" in node.last_sensor_error


def test_bad_right_target_prevents_both_arms_moving(ros_logic):
    node = ros_logic.E03Node.__new__(ros_logic.E03Node)
    node.config = configured_robot()
    node.gripper_clients = [None, None]
    node.move_client = NS(wait_for_server=lambda **kwargs: True)
    node.observation_age = lambda _: 0.1
    node.current_pose = lambda _: np.eye(4)
    sent = []
    node._actions = lambda requests: sent.append(requests)
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    targets[1, 0, 3] = 0.5
    with pytest.raises(ValueError, match="translation"):
        node.execute_target(targets, [0, 1], 0, plan_only=False)
    assert sent == []


def test_plan_only_never_sends_gripper_commands(ros_logic):
    node = ros_logic.E03Node.__new__(ros_logic.E03Node)
    node.config = configured_robot()
    node.gripper_clients = [object(), object()]
    node.move_client = NS(wait_for_server=lambda **kwargs: True)
    node.observation_age = lambda _: 0.1
    node.current_pose = lambda _: np.eye(4)
    sent = []

    def action(requests):
        sent.extend(requests)
        return [NS(error_code=NS(val=1))]

    node._actions = action
    node.execute_target(np.repeat(np.eye(4)[None], 2, axis=0), [0, 1], 0, plan_only=True)
    assert len(sent) == 1 and sent[0][1].planning_options.plan_only


def ready(value):
    future = Future()
    future.set_result(value)
    return future


def test_parallel_action_failure_cancels_the_other_gripper(ros_logic):
    node = ros_logic.E03Node.__new__(ros_logic.E03Node)
    node.config = configured_robot()
    node.active_goals, node.pending_goals = [], []
    node.get_logger = lambda: NS(error=lambda _: None)
    canceled, sent = [], []
    handles = []
    for index in range(2):
        handles.append(NS(accepted=True,
                          get_result_async=lambda index=index: ready(NS(status=6 if index == 0 else 4, result=None)),
                          cancel_goal_async=lambda index=index: canceled.append(index) or ready(NS(goals_canceling=[1]))))
    clients = [NS(wait_for_server=lambda **kwargs: True,
                  send_goal_async=lambda goal, index=index: sent.append(index) or ready(handles[index]))
               for index in range(2)]
    with pytest.raises(RuntimeError, match="status=6"):
        node._actions([(clients[0], object()), (clients[1], object())])
    assert sent == [0, 1] and canceled == [1]
    assert node.active_goals == [] and node.pending_goals == []
