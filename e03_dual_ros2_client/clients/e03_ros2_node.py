"""ROS2 sensor bridge and coordinated MoveIt2 execution for dual E03 arms.

Import this module only in a sourced ROS2 environment (Humble or newer).
No hardware enabling, homing, controller switching or fault clearing is done.
"""

import collections
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, MoveItErrorCodes, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from .e03_protocol import check_target, observation_frame, remap_gripper, transform_matrix


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_message(matrix):
    matrix = transform_matrix(matrix)
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = map(float, matrix[:3, 3])
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = map(float, quaternion)
    return msg


def move_goal(target, config, plan_only):
    """One combined planning group handles both arm targets and self-collision."""
    goal = MoveGroup.Goal()
    request = goal.request
    motion = config["motion"]
    request.group_name = config["planning_group"]
    request.num_planning_attempts = 1
    request.allowed_planning_time = float(motion["planning_time_s"])
    request.max_velocity_scaling_factor = float(motion["velocity_scale"])
    request.max_acceleration_scaling_factor = float(motion["acceleration_scale"])
    request.start_state.is_diff = True  # Use MoveIt's current complete robot state.

    if len(target) != len(config["arms"]):
        raise ValueError("Every arm must have a target in the combined MoveIt goal")
    constraint = Constraints()
    for pose, arm in zip(target, config["arms"]):
        position = PositionConstraint()
        position.header.frame_id = config["reference_frame"]
        position.link_name = arm["tcp_frame"]
        position.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(motion["position_tolerance_m"])]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [pose_message(pose)]
        constraint.position_constraints.append(position)

        orientation = OrientationConstraint()
        orientation.header.frame_id = config["reference_frame"]
        orientation.link_name = arm["tcp_frame"]
        orientation.orientation = pose_message(pose).orientation
        orientation.absolute_x_axis_tolerance = float(motion["orientation_tolerance_rad"])
        orientation.absolute_y_axis_tolerance = float(motion["orientation_tolerance_rad"])
        orientation.absolute_z_axis_tolerance = float(motion["orientation_tolerance_rad"])
        orientation.weight = 1.0
        constraint.orientation_constraints.append(orientation)
    # A single Constraints object means AND. Multiple goal_constraints entries
    # would mean alternative goals and could allow only one arm to reach target.
    request.goal_constraints = [constraint]
    goal.planning_options.plan_only = plan_only
    goal.planning_options.replan = False
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


class E03Node(Node):
    def __init__(self, config, period, history_seconds):
        super().__init__("e03_e2vla_client")
        self.config = config
        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=max(10.0, history_seconds + 2)))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.images, self.intrinsics, self.joints = {}, {}, {}
        self.origin_ns = None
        self.last_stamp_ns = -1
        self.sequence = 0
        self.last_sensor_error = "Waiting for ROS images, CameraInfo, JointState and TF"
        self.capture_lock = threading.Lock()
        self.frames = collections.deque(maxlen=max(8, int((history_seconds + 2) / period) + 2))
        self.active_goals = []
        self.pending_goals = []
        self.stopping = False
        self.subscriptions_owned = []
        for topic in config["joint_states_topics"]:
            self.subscriptions_owned.append(self.create_subscription(
                JointState, topic, self._joint_callback, qos_profile_sensor_data,
            ))
        for name, camera in config["cameras"].items():
            self.subscriptions_owned.append(self.create_subscription(
                Image, camera["image_topic"],
                lambda msg, name=name: self.images.__setitem__(name, msg), qos_profile_sensor_data,
            ))
            self.subscriptions_owned.append(self.create_subscription(
                CameraInfo, camera["camera_info_topic"],
                lambda msg, name=name: self.intrinsics.__setitem__(name, msg), qos_profile_sensor_data,
            ))
        self.capture_timer = self.create_timer(period, self._capture)
        self.move_client = ActionClient(self, MoveGroup, config["move_group_action"])
        self.gripper_clients = [
            ActionClient(self, GripperCommand, arm["gripper"]["action_name"])
            if arm["gripper"]["mode"] == "action" else None
            for arm in config["arms"]
        ]

    def _joint_callback(self, msg):
        # Some drivers publish arm and gripper joints in separate messages.
        timestamp = stamp_ns(msg.header.stamp)
        for name, value in zip(msg.name, msg.position):
            if np.isfinite(value):
                self.joints[name] = (float(value), timestamp)

    def _tf_matrix(self, child, when):
        parent = self.config["reference_frame"]
        transform = self.tf_buffer.lookup_transform(parent, child, when)
        t, q = transform.transform.translation, transform.transform.rotation
        matrix = np.eye(4)
        matrix[:3, 3] = [t.x, t.y, t.z]
        matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return matrix, stamp_ns(transform.header.stamp)

    def _model_gripper(self, arm):
        gripper = arm["gripper"]
        if gripper["mode"] == "fixed":
            return float(gripper["fixed_model_value"])
        raw, _ = self.joints[gripper["joint_name"]]
        return remap_gripper(raw, gripper["state_closed"], gripper["state_open"],
                            gripper["model_closed"], gripper["model_open"])

    def _capture(self):
        if self.stopping:
            return
        try:
            stamps = [self.joints[name][1] for arm in self.config["arms"] for name in arm["joint_names"]]
            for arm in self.config["arms"]:
                if arm["gripper"]["mode"] == "action":
                    stamps.append(self.joints[arm["gripper"]["joint_name"]][1])
            stamps += [stamp_ns(self.images[name].header.stamp) for name in self.config["cameras"]]
            now_ns = self.get_clock().now().nanoseconds
            if min(stamps) <= 0:
                raise ValueError("Sensors must publish nonzero ROS header timestamps")
            if (now_ns - min(stamps)) / 1e9 > self.config["max_sensor_age_s"]:
                raise ValueError("Image or joint feedback is stale")
            if (max(stamps) - now_ns) / 1e9 > self.config["max_sensor_skew_s"]:
                raise ValueError("Sensor timestamps are ahead of ROS clock; check clock synchronization")
            if (max(stamps) - min(stamps)) / 1e9 > self.config["max_sensor_skew_s"]:
                raise ValueError("Images and joint feedback are not synchronized within max_sensor_skew_s")
            # Use the oldest contributing timestamp; TF is queried at sensor time,
            # including wrist camera transforms, never replaced with identity.
            anchor = min(stamps)
            if anchor <= self.last_stamp_ns:
                if anchor < self.last_stamp_ns:
                    raise ValueError("ROS time moved backwards; restart the client to begin a new model history")
                return
            ee_poses = [self._tf_matrix(arm["tcp_frame"], Time(nanoseconds=anchor))[0]
                        for arm in self.config["arms"]]
            origin = anchor if self.origin_ns is None else self.origin_ns
            cameras = {}
            for name, camera in self.config["cameras"].items():
                image, info = self.images[name], self.intrinsics[name]
                if (info.width, info.height) != (image.width, image.height):
                    raise ValueError(f"{name}: CameraInfo resolution differs from image")
                if not image.header.frame_id or image.header.frame_id != info.header.frame_id:
                    raise ValueError(f"{name}: image and CameraInfo must identify the same optical frame")
                if camera["rectified"]:
                    K = np.asarray(info.p, dtype=float).reshape(3, 4)[:, :3]
                else:
                    if np.any(np.abs(info.d) > 1e-8):
                        raise ValueError(f"{name}: use rectified image/CameraInfo for a distorted camera")
                    K = np.asarray(info.k, dtype=float).reshape(3, 3)
                image_stamp = stamp_ns(image.header.stamp)
                wcT, _ = self._tf_matrix(image.header.frame_id, Time(nanoseconds=image_stamp))
                if camera["rectified"]:
                    # CameraInfo.R maps original optical coordinates into the
                    # rectified optical frame used by P and the rectified image.
                    rectification = np.eye(4)
                    rectification[:3, :3] = np.asarray(info.r, dtype=float).reshape(3, 3).T
                    wcT = wcT @ transform_matrix(rectification, f"{name}.rectification")
                cameras[name] = {
                    "rgb": self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8").copy(),
                    "K": K, "wcT": wcT, "timestamp": (image_stamp - origin) / 1e9,
                }
            grippers = [self._model_gripper(arm) for arm in self.config["arms"]]
            frame = observation_frame(ee_poses, grippers, (anchor - origin) / 1e9, cameras)
            with self.capture_lock:
                self.origin_ns = origin
                self.last_stamp_ns = anchor
                self.sequence += 1
                self.frames.append((self.sequence, time.monotonic(), frame))
                self.last_sensor_error = ""
        except Exception as exc:
            # TF may arrive after the sensor message. Keep sampling until it does;
            # the main thread enforces a finite observation timeout.
            with self.capture_lock:
                self.last_sensor_error = f"{type(exc).__name__}: {exc}"

    def wait_frames(self, after_sequence, history_seconds=0.0):
        deadline = time.monotonic() + self.config["observation_timeout_s"]
        while rclpy.ok() and time.monotonic() < deadline:
            with self.capture_lock:
                rows = [row for row in self.frames if row[0] > after_sequence]
                if rows:
                    span = rows[-1][2]["timestamp"] - rows[0][2]["timestamp"]
                    fresh = time.monotonic() - rows[-1][1] <= self.config["max_sensor_age_s"]
                    if fresh and span + 1e-6 >= history_seconds:
                        return rows[-1][0], [row[2] for row in rows]
                error = self.last_sensor_error
            time.sleep(0.02)
        raise TimeoutError(f"Timed out waiting for observations/history: {error or 'insufficient fresh history'}")

    def current_pose(self, arm):
        matrix, timestamp = self._tf_matrix(arm["tcp_frame"], Time())
        age = (self.get_clock().now().nanoseconds - timestamp) / 1e9
        if timestamp <= 0 or age > self.config["max_sensor_age_s"] or age < -self.config["max_sensor_skew_s"]:
            raise ValueError("Current TCP TF is stale or has an invalid timestamp")
        return matrix

    def observation_age(self, model_timestamp):
        if self.origin_ns is None:
            raise ValueError("No observations captured")
        return (self.get_clock().now().nanoseconds - self.origin_ns) / 1e9 - model_timestamp

    @staticmethod
    def _wait(future, timeout):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise TimeoutError("ROS action request/result timed out")
        return future.result()

    def _actions(self, requests):
        """Submit together, wait for all; on error cancel all unfinished goals."""
        timeout = self.config["motion"]["action_timeout_s"]
        for client, _ in requests:
            if not client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError("ROS action server unavailable; check configured action names")
        pending, handles, completed = [], [], []
        try:
            for client, goal in requests:
                future = client.send_goal_async(goal)
                pending.append(future)
                self.pending_goals.append(future)
            for future in pending:
                handle = self._wait(future, timeout)
                handles.append(handle)
                if not handle.accepted:
                    raise RuntimeError("ROS action goal was rejected")
                self.active_goals.append(handle)
            results = []
            for handle in handles:
                result = self._wait(handle.get_result_async(), timeout)
                completed.append(handle)
                if result.status != GoalStatus.STATUS_SUCCEEDED:
                    raise RuntimeError(f"ROS action failed with status={result.status}")
                results.append(result.result)
            return results
        except BaseException:
            for handle in handles:
                if handle.accepted and handle not in completed and self._cancel(handle):
                    completed.append(handle)
            for future in pending[len(handles):]:
                # Even an acceptance arriving after a timeout must be canceled.
                def cancel_late(future):
                    try:
                        late = future.result()
                        if late is not None and late.accepted:
                            late.cancel_goal_async()
                    except Exception as exc:
                        self.get_logger().error(f"Could not cancel late goal: {exc}")
                future.add_done_callback(cancel_late)
            raise
        finally:
            for future, handle in zip(pending, handles):
                self.pending_goals.remove(future)
            for handle in completed:
                if handle in self.active_goals:
                    self.active_goals.remove(handle)

    def _cancel(self, handle):
        try:
            response = self._wait(handle.cancel_goal_async(), 3.0)
            if not response.goals_canceling:
                self.get_logger().error("Controller did not confirm goal cancellation")
            return bool(response.goals_canceling)
        except Exception as exc:
            self.get_logger().error(f"Goal cancellation failed: {exc}")
            return False

    def execute_target(self, poses, gripper_values, observation_time, plan_only):
        if len(poses) != len(self.config["arms"]) or len(gripper_values) != len(poses):
            raise ValueError("Every arm needs a pose and gripper target")
        # Discover all needed action servers before validating prediction age.
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("MoveIt move_group action server is unavailable")
        if not plan_only:
            for client in self.gripper_clients:
                if client is not None and not client.wait_for_server(timeout_sec=5.0):
                    raise RuntimeError("Configured gripper action server is unavailable")
        prediction_age = self.observation_age(observation_time)
        if prediction_age < -self.config["max_sensor_skew_s"]:
            raise ValueError("ROS time moved backwards after inference")
        if prediction_age > self.config["motion"]["max_prediction_age_s"]:
            raise ValueError("Prediction is too old to execute; increase timeout only after measuring latency")
        for pose, arm in zip(poses, self.config["arms"]):
            limits = dict(self.config["motion"], **arm["workspace"])
            check_target(self.current_pose(arm), pose, limits)
        result = self._actions([(self.move_client, move_goal(poses, self.config, plan_only))])[0]
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt failed with error_code={result.error_code.val}")
        if plan_only:
            return
        requests, gripper_configs = [], []
        for value, arm, client in zip(gripper_values, self.config["arms"], self.gripper_clients):
            if client is None:
                continue
            gripper = arm["gripper"]
            goal = GripperCommand.Goal()
            goal.command.position = remap_gripper(
                value, gripper["model_closed"], gripper["model_open"],
                gripper["command_closed_m"], gripper["command_open_m"],
            )
            goal.command.max_effort = float(gripper["max_effort_n"])
            requests.append((client, goal))
            gripper_configs.append(gripper)
        for result, gripper in zip(self._actions(requests), gripper_configs):
            if not result.reached_goal and not (result.stalled and gripper["allow_stall"]):
                raise RuntimeError("Gripper did not reach its goal")

    def stop(self):
        self.stopping = True
        self.capture_timer.cancel()
        for handle in list(self.active_goals):
            self._cancel(handle)
        for pending in self.pending_goals:
            try:
                # Give late acceptance/cancellation callbacks time to complete
                # while the executor is still running.
                handle = self._wait(pending, 3.0)
                if handle.accepted:
                    self._cancel(handle)
            except Exception as exc:
                self.get_logger().error(f"Pending goal status could not be confirmed: {exc}")
