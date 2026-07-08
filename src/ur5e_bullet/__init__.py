import os
import math
import random
import time
import numpy as np
import pybullet
import pybullet_data
from collections import namedtuple

from pybullet_planning import plan_joint_motion

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_URDF_PATH = os.path.join(_PKG_DIR, "ur_e_description", "urdf", "ur5e.urdf")
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")


class UR5Sim():

    def __init__(self, gui=True):
        pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        pybullet.setRealTimeSimulation(True)

        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)

        self.control_joints = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.joint_names = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", [
            "id", "name", "type", "lowerLimit", "upperLimit",
            "maxForce", "maxVelocity", "controllable",
        ])
        self.joints = {}
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            j = self.joint_info(
                info[0], info[1].decode("utf-8"),
                self.joint_names[info[2]], info[8], info[9],
                info[10], info[11], info[1].decode("utf-8") in self.control_joints,
            )
            if j.type == "REVOLUTE":
                pybullet.setJointMotorControl2(
                    self.ur5, j.id, pybullet.VELOCITY_CONTROL,
                    targetVelocity=0, force=0,
                )
            self.joints[j.name] = j

        self._joint_ids = [self.joints[n].id for n in self.control_joints]
        self._null_space = (
            [-math.pi]*6, [math.pi]*6, [2*math.pi]*6,
            [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0],
        )
        self._ik_lambda = 0.05

        self.tool_offset_pos = None
        self.tool_offset_orn = None
        self._last_conf = None
        self._last_target = None

    def load_robot(self):
        self._table = pybullet.loadURDF(
            TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1],
        )
        return pybullet.loadURDF(
            ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1],
            flags=pybullet.URDF_USE_SELF_COLLISION,
        )

    def set_joint_angles(self, joint_angles):
        pybullet.setJointMotorControlArray(
            self.ur5, self._joint_ids,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*6,
            positionGains=[0.04]*6,
            forces=[self.joints[n].maxForce for n in self.control_joints],
        )

    def get_joint_angles(self):
        return [s[0] for s in pybullet.getJointStates(self.ur5, [1, 2, 3, 4, 5, 6])]

    def _ik(self, ee_pose, seed=None, max_iter=10, tol=1e-6):
        target_pos, target_quat = ee_pose
        rest = seed if seed is not None else self._null_space[3]
        conf = list(pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, target_pos, target_quat,
            lowerLimits=self._null_space[0], upperLimits=self._null_space[1],
            jointRanges=self._null_space[2], restPoses=rest,
            solver=pybullet.IK_DLS,
        ))
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, v in zip(self._joint_ids, conf):
            pybullet.resetJointState(self.ur5, i, v)

        for _ in range(max_iter):
            ls = pybullet.getLinkState(
                self.ur5, self.end_effector_index,
                computeForwardKinematics=True,
            )
            pos_err = [target_pos[i] - ls[0][i] for i in range(3)]
            dx = math.sqrt(sum(e**2 for e in pos_err))
            if dx <= tol:
                break

            q_conj = [-ls[1][0], -ls[1][1], -ls[1][2], ls[1][3]]
            q_err = pybullet.multiplyTransforms([0,0,0], target_quat, [0,0,0], q_conj)[1]
            angle = 2 * math.acos(max(-1, min(1, q_err[3])))
            if angle > 1e-10:
                s = math.sin(angle / 2)
                ori_err = [q_err[i] * angle / s for i in range(3)]
            else:
                ori_err = [0, 0, 0]

            jac = pybullet.calculateJacobian(
                self.ur5, self.end_effector_index, [0,0,0],
                list(conf), [0]*6, [0]*6,
            )
            J = np.vstack([np.array(jac[0]), np.array(jac[1])])
            err = np.array(pos_err + ori_err)
            lam = self._ik_lambda
            delta = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            conf = [conf[i] + delta[i] for i in range(6)]
            for i, v in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, i, v)

        for i, v in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, i, v)
        return conf

    def _tcp_to_ee(self, pos, ori):
        quat = pybullet.getQuaternionFromEuler(ori)
        if self.tool_offset_pos is None:
            return (pos, quat)
        q_inv = [self.tool_offset_orn[0], -self.tool_offset_orn[1],
                 -self.tool_offset_orn[2], -self.tool_offset_orn[3]]
        q_ee = pybullet.multiplyTransforms([0,0,0], quat, [0,0,0], q_inv)[1]
        R = np.array(pybullet.getMatrixFromQuaternion(q_ee)).reshape(3, 3)
        pos_ee = [pos[i] - (R @ np.array(self.tool_offset_pos))[i] for i in range(3)]
        return (pos_ee, q_ee)

    def _conf_for(self, ee_pose, seed=None):
        conf = self._ik(ee_pose, seed=seed)
        if conf is None:
            return None
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, val in zip(self._joint_ids, conf):
            pybullet.resetJointState(self.ur5, i, val)
        ls = pybullet.getLinkState(
            self.ur5, self.end_effector_index,
            computeForwardKinematics=True,
        )
        for i, val in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, i, val)
        dx = math.sqrt(sum((ls[0][i] - ee_pose[0][i])**2 for i in range(3)))
        if dx > 0.08:
            return None
        return conf

    def _find_collision_free_conf(self, ee_target):
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        strategies = [
            None,
            [0, -math.pi/2, 0, 0, 0, 0],
            [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0],
            [math.pi, -math.pi/2, math.pi/2, 0, -math.pi/2, 0],
            [math.pi, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0],
            [0, -math.pi/4, 0, 0, 0, 0],
            [0, -math.pi/3, 0, 0, 0, 0],
            [math.pi/2, -math.pi/2, 0, 0, 0, 0],
            [-math.pi/2, -math.pi/2, 0, 0, 0, 0],
            [0, -math.pi/2, -1.0, -math.pi/2, -math.pi/2, 0],
        ]
        for pose in strategies:
            if pose is not None:
                for j, v in zip(self._joint_ids, pose):
                    pybullet.resetJointState(self.ur5, j, v)
            conf = self._conf_for(ee_target)
            if conf is not None and self._collision_free(conf):
                for j, v in zip(self._joint_ids, saved):
                    pybullet.resetJointState(self.ur5, j, v)
                return conf
        for j, v in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, j, v)
        return None

    def _execute(self, path, speed=1.0):
        for conf in path:
            for i, val in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, i, val)
            pybullet.stepSimulation()
            if speed > 0:
                time.sleep(0.01 / speed)

    def _linear_segment(self, ee_start, ee_end):
        steps = 100
        path = []
        initial_state = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i in range(1, steps + 1):
            t = i / steps
            pos = [
                ee_start[0][j] + t * (ee_end[0][j] - ee_start[0][j])
                for j in range(3)
            ]
            quat = pybullet.getQuaternionSlerp(ee_start[1], ee_end[1], t)
            for j, v in zip(self._joint_ids, path[-1] if path else initial_state):
                pybullet.resetJointState(self.ur5, j, v)
            conf = self._ik((pos, quat))
            if conf is None or not self._collision_free(conf):
                for j, v in zip(self._joint_ids, initial_state):
                    pybullet.resetJointState(self.ur5, j, v)
                return None
            path.append(conf)
        for j, v in zip(self._joint_ids, initial_state):
            pybullet.resetJointState(self.ur5, j, v)
        return path

    def _probe_rrt(self, target_pos, target_ori):
        start_tcp, _ = self.get_tcp_pose()
        original_joint_positions = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        ee_target = self._tcp_to_ee(target_pos, target_ori)
        render_flag = pybullet.COV_ENABLE_RENDERING
        pybullet.configureDebugVisualizer(render_flag, 0)

        target_configuration = self._conf_for(ee_target)
        if target_configuration is None or not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
        if target_configuration is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            for j, v in zip(self._joint_ids, original_joint_positions):
                pybullet.resetJointState(self.ur5, j, v)
            return start_tcp, target_pos, 0.0, target_pos, None

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        path = plan_joint_motion(
            self.ur5, self._joint_ids, target_configuration,
            obstacles=[], self_collisions=True,
            restarts=10, smooth=30,
        )
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)

        if path is not None:
            tcp_waypoints = []
            for configuration in path:
                for j, v in zip(self._joint_ids, configuration):
                    pybullet.resetJointState(self.ur5, j, v)
                tcp, _ = self.get_tcp_pose()
                tcp_waypoints.append(list(tcp))
            for j, v in zip(self._joint_ids, original_joint_positions):
                pybullet.resetJointState(self.ur5, j, v)
            pybullet.configureDebugVisualizer(render_flag, 1)
            return target_pos, None, 0.0, target_pos, tcp_waypoints

        for j, v in zip(self._joint_ids, original_joint_positions):
            pybullet.resetJointState(self.ur5, j, v)
        pybullet.configureDebugVisualizer(render_flag, 1)
        return start_tcp, target_pos, 0.0, target_pos, None

    def _eval_linear_path(self, path, start_tcp, target_pos):
        num_steps = len(path)
        original_joint_positions = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        last_good_fraction = 0.0

        for i, configuration in enumerate(path):
            fraction = (i + 1) / num_steps
            target_tcp_at_t = [
                start_tcp[j] + fraction * (target_pos[j] - start_tcp[j])
                for j in range(3)
            ]
            for j, v in zip(self._joint_ids, configuration):
                pybullet.resetJointState(self.ur5, j, v)
            actual_tcp, _ = self.get_tcp_pose()
            deviation_mm = math.sqrt(sum((actual_tcp[j]-target_tcp_at_t[j])**2 for j in range(3))) * 1000
            if deviation_mm <= 5:
                last_good_fraction = fraction

        last_good_position = [
            start_tcp[j] + last_good_fraction * (target_pos[j] - start_tcp[j])
            for j in range(3)
        ]
        configuration = path[-1]
        for j, v in zip(self._joint_ids, configuration):
            pybullet.resetJointState(self.ur5, j, v)
        actual_endpoint, _ = self.get_tcp_pose()
        end_deviation_mm = math.sqrt(sum((actual_endpoint[j]-target_pos[j])**2 for j in range(3))) * 1000

        for j, v in zip(self._joint_ids, original_joint_positions):
            pybullet.resetJointState(self.ur5, j, v)

        if last_good_fraction >= 1.0 and end_deviation_mm <= 5:
            return target_pos, None, 0.0, actual_endpoint, None
        return last_good_position, None, end_deviation_mm, actual_endpoint, None

    def _probe_linear_stepwise(self, ee_start, ee_target, start_tcp):
        num_steps = 100
        seed_configuration = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        last_valid_position = ee_start[0]

        for i in range(1, num_steps + 1):
            fraction = i / num_steps
            target_position_at_t = [
                ee_start[0][j] + fraction * (ee_target[0][j] - ee_start[0][j])
                for j in range(3)
            ]
            interpolated_orientation = pybullet.getQuaternionSlerp(ee_start[1], ee_target[1], fraction)
            configuration = self._ik((target_position_at_t, interpolated_orientation), seed=seed_configuration)
            if configuration is None or not self._collision_free(configuration):
                return start_tcp, target_position_at_t, 0.0, start_tcp, None
            seed_configuration = configuration
            last_valid_position = target_position_at_t

        return last_valid_position, None, 0.0, last_valid_position, None

    def _probe_linear(self, target_pos, target_ori):
        ee_target = self._tcp_to_ee(target_pos, target_ori)
        ee_start = pybullet.getLinkState(
            self.ur5, self.end_effector_index,
            computeForwardKinematics=True,
        )[:2]
        start_tcp, _ = self.get_tcp_pose()
        render_flag = pybullet.COV_ENABLE_RENDERING
        pybullet.configureDebugVisualizer(render_flag, 0)

        path = self._linear_segment(ee_start, ee_target)
        if path is not None:
            result = self._eval_linear_path(path, start_tcp, target_pos)
            pybullet.configureDebugVisualizer(render_flag, 1)
            return result

        result = self._probe_linear_stepwise(ee_start, ee_target, start_tcp)
        pybullet.configureDebugVisualizer(render_flag, 1)
        return result

    def _probe_path(self, target_pos, target_ori, rrt=False):
        if rrt:
            return self._probe_rrt(target_pos, target_ori)
        return self._probe_linear(target_pos, target_ori)

    def _collision_free(self, configuration):
        original_joint_positions = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, v in zip(self._joint_ids, configuration):
            pybullet.resetJointState(self.ur5, i, v)
        collision_free = True
        for i in range(-1, self.num_joints):
            for j in range(i+1, self.num_joints):
                if abs(i - j) <= 1:
                    continue
                contacts = pybullet.getClosestPoints(
                    self.ur5, self.ur5, 0.0,
                    linkIndexA=i, linkIndexB=j,
                )
                if contacts:
                    collision_free = False
                    break
            if not collision_free:
                break
        if collision_free and hasattr(self, '_table'):
            for i in range(-1, self.num_joints):
                contacts = pybullet.getClosestPoints(
                    self.ur5, self._table, 0.0,
                    linkIndexA=i,
                )
                if contacts:
                    collision_free = False
                    break
        for i, v in zip(self._joint_ids, original_joint_positions):
            pybullet.resetJointState(self.ur5, i, v)
        return collision_free

    def _via_point(self, start_pos, target_pos, lift=0.1):
        z = max(start_pos[2], target_pos[2]) + lift
        return [(start_pos[0], start_pos[1], z),
                (target_pos[0], target_pos[1], z)]

    def _find_path(self, ee_start, ee_target):
        """Direct linear path (2 seed variants) → None."""
        render_flag = pybullet.COV_ENABLE_RENDERING
        pybullet.configureDebugVisualizer(render_flag, 0)
        path = self._linear_segment(ee_start, ee_target)
        if path is not None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            return path
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for j, v in zip(self._joint_ids, self._null_space[3]):
            pybullet.resetJointState(self.ur5, j, v)
        path = self._linear_segment(ee_start, ee_target)
        for j, v in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, j, v)
        pybullet.configureDebugVisualizer(render_flag, 1)
        return path

    def move_to(self, tcp_pos, tcp_ori, linear=True, obstacles=[], tol=0.08, speed=1.0):
        ee_target = self._tcp_to_ee(tcp_pos, tcp_ori)
        target_configuration = self._conf_for(ee_target)
        if target_configuration is None:
            print(f"[nein] Ziel {tcp_pos} nicht erreichbar")
            return False

        def run(path, label=""):
            self._execute(path, speed)
            actual_tcp, _ = self.get_tcp_pose()
            error_mm = math.sqrt(sum((actual_tcp[i]-tcp_pos[i])**2 for i in range(3)))
            if error_mm > 0.005:
                print(f"[!] {tcp_pos} → {error_mm*1000:.0f} mm daneben")
            else:
                print(f"[ok] {tcp_pos}{label}")
            self._save_last(tcp_pos, tcp_ori)
            return True

        if linear:
            ee_start = pybullet.getLinkState(
                self.ur5, self.end_effector_index,
                computeForwardKinematics=True,
            )[:2]
            path = self._find_path(ee_start, ee_target)
            if path is not None:
                return run(path)

            print(f"[nein] Kein Pfad zu {tcp_pos}")
            return False

        if not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
            if target_configuration is None:
                print(f"[nein] Ziel {tcp_pos} kollidiert")
                return False
        path = plan_joint_motion(
            self.ur5, self._joint_ids, target_configuration,
            obstacles=obstacles, self_collisions=True,
            restarts=10, smooth=30,
        )
        if path is None:
            print(f"[nein] Kein Pfad zu {tcp_pos}")
            return False
        return run(path)

    def _save_last(self, tcp_pos, tcp_ori):
        self._last_conf = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        self._last_target = (tcp_pos, tcp_ori)

    def get_current_pose(self):
        ls = pybullet.getLinkState(
            self.ur5, self.end_effector_index, computeForwardKinematics=True
        )
        return ls[0], ls[1]

    def set_tool_offset(self, pos, ori=(0, 0, 0, 1)):
        self.tool_offset_pos = list(pos)
        self.tool_offset_orn = list(ori)

    def clear_tool_offset(self):
        self.tool_offset_pos = None
        self.tool_offset_orn = None

    def get_tcp_pose(self):
        pos, quat = self.get_current_pose()
        if self.tool_offset_pos is not None:
            R = np.array(pybullet.getMatrixFromQuaternion(quat)).reshape(3, 3)
            pos = [pos[i] + (R @ np.array(self.tool_offset_pos))[i] for i in range(3)]
            quat = pybullet.multiplyTransforms(
                [0, 0, 0], quat, self.tool_offset_pos, self.tool_offset_orn
            )[1]
        return pos, quat


def _draw_crosshair(pos, color, items, label=None):
    h = 0.03
    items.append(pybullet.addUserDebugLine(
        [pos[0]-h, pos[1], pos[2]], [pos[0]+h, pos[1], pos[2]], color, 2,
    ))
    items.append(pybullet.addUserDebugLine(
        [pos[0], pos[1]-h, pos[2]], [pos[0], pos[1]+h, pos[2]], color, 2,
    ))
    items.append(pybullet.addUserDebugLine(
        [pos[0], pos[1], pos[2]-h], [pos[0], pos[1], pos[2]+h], color, 2,
    ))
    if label:
        items.append(pybullet.addUserDebugText(
            label, [pos[0], pos[1], pos[2]+0.04], color, 0.8,
        ))
    return items


Command = namedtuple("Command", [
    "action", "params",
])


def _parse_command(tokens):
    if tokens[0] == "q":
        return Command("quit", {})
    if tokens[0] == "o" and len(tokens) == 4:
        try:
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        except ValueError:
            print(f"  ? '{' '.join(tokens)}' verstanden?")
            return Command("error", {})
        return Command("offset", {"pos": pos})
    if tokens[0] == "s" and len(tokens) == 2:
        try:
            speed = float(tokens[1])
        except ValueError:
            print(f"  ? '{' '.join(tokens)}' verstanden?")
            return Command("error", {})
        return Command("speed", {"speed": max(speed, 0.01)})
    if tokens[0] == "@":
        return Command("reset", {})

    linear = True
    idx = 0
    if tokens[0] == "r":
        linear = False
        idx = 1
    try:
        parsed_values = [float(v) for v in tokens[idx:]]
    except ValueError:
        print(f"  ? '{' '.join(tokens)}' verstanden?")
        return Command("error", {})
    if len(parsed_values) < 3:
        return Command("incomplete", {})
    target_position = parsed_values[:3]
    target_orientation = [math.radians(v) for v in parsed_values[3:6]] if len(parsed_values) >= 6 else [0, 0, 0]
    return Command("move", {
        "target_position": target_position,
        "target_orientation": target_orientation,
        "linear": linear,
    })


def demo_simulation():
    sim = UR5Sim()

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

    def draw_probe_preview(last_valid_position, collision_position, deviation_mm,
                           rrt_waypoints, start_tcp, target_position, target_orientation,
                           linear):
        if collision_position is None and deviation_mm == 0.0:
            if rrt_waypoints:
                step = max(1, len(rrt_waypoints) // 20)
                for i in range(0, len(rrt_waypoints)-step, step):
                    items.append(pybullet.addUserDebugLine(
                        rrt_waypoints[i], rrt_waypoints[i+step], [0, 1, 0], 1,
                    ))
            else:
                items.append(pybullet.addUserDebugLine(
                    start_tcp, target_position, [0, 1, 0], 2,
                ))
            return target_position, target_orientation

        if collision_position is None and deviation_mm > 0:
            items.append(pybullet.addUserDebugLine(
                start_tcp, last_valid_position, [0, 1, 0], 2,
            ))
            items.append(pybullet.addUserDebugLine(
                last_valid_position, target_position, [1, 0, 0], 2,
            ))
            reachable_distance = math.sqrt(sum((last_valid_position[i]-start_tcp[i])**2 for i in range(3)))
            if reachable_distance > 0.005:
                target_distance = math.sqrt(sum((target_position[i]-start_tcp[i])**2 for i in range(3)))
                slerp_fraction = reachable_distance / target_distance if target_distance > 0 else 0
                _, current_orientation = sim.get_tcp_pose()
                mid_orientation = pybullet.getQuaternionSlerp(
                    current_orientation, pybullet.getQuaternionFromEuler(target_orientation), slerp_fraction,
                )
                print(f"  ⚠ {deviation_mm:.0f}mm Abweichung – fahre zu ({last_valid_position[0]:.3f}, {last_valid_position[1]:.3f}, {last_valid_position[2]:.3f})")
                return last_valid_position, list(pybullet.getEulerFromQuaternion(mid_orientation))
            else:
                _draw_crosshair(target_position, [1, 0, 0], items)
                print(f"  ⛔ Kein linearer Pfad zu ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
                return target_position, target_orientation

        items.append(pybullet.addUserDebugLine(
            start_tcp, last_valid_position, [0, 1, 0], 2,
        ))
        items.append(pybullet.addUserDebugLine(
            last_valid_position, target_position, [1, 0, 0], 2,
        ))
        _draw_crosshair(collision_position, [1, 0, 0], items)
        if not linear:
            print(f"  ⛔ Kein RRT-Pfad zu ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
        else:
            print(f"  ⛔ Kollision bei ({collision_position[0]:.3f}, {collision_position[1]:.3f}, {collision_position[2]:.3f})")
        return target_position, target_orientation

    print("── UR5e Demo ──────────────────────────────")
    print("Format: x y z rx ry rz oder 'q' zum Beenden")
    print("  Tool-Offset: 'o x y z' (z. B. o 0 0 0.15)")
    print("  RRT-Modus:   'r x y z rx ry rz'")
    print("  Geschw.:     's 0.5' (global, Default 0.5)")
    print("  Reset:       '@'  (nach manuellem Ziehen)")
    print("────────────────────────────────────────────")
    items = []
    current_speed = 0.5
    tcp_items = draw_tcp()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        tokens = line.split()
        cmd = _parse_command(tokens)
        if cmd.action == "quit":
            break
        if cmd.action == "offset":
            sim.set_tool_offset(cmd.params["pos"])
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            continue
        if cmd.action == "reset" and sim._last_conf is not None:
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            sim._execute([sim._last_conf])
            tcp_items = draw_tcp()
            continue
        if cmd.action == "speed":
            current_speed = cmd.params["speed"]
            print(f"  Geschwindigkeit: {current_speed:.2f}")
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            continue

        pybullet.removeAllUserDebugItems()
        items.clear()
        tcp_items = []

        if cmd.action == "error":
            continue
        if cmd.action == "incomplete":
            tcp_items = draw_tcp()
            if sim._last_target:
                tcp_pos, _ = sim.get_tcp_pose()
                displacement_mm = math.sqrt(sum((tcp_pos[i]-sim._last_target[0][i])**2 for i in range(3)))
                if displacement_mm > 0.01:
                    print(f"  ⚠ {displacement_mm*1000:.0f}mm manuell verschoben (@ zum Reset)")
            continue

        target_position = cmd.params["target_position"]
        target_orientation = cmd.params["target_orientation"]
        linear = cmd.params["linear"]

        tcp_pos, _ = sim.get_tcp_pose()
        _draw_crosshair(target_position, [1, 1, 0], items,
                        f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

        probe_result = sim._probe_path(target_position, target_orientation, rrt=not linear)
        last_valid_position, collision_position, deviation_mm, actual_endpoint, rrt_waypoints = probe_result

        target_pos, target_ori = draw_probe_preview(
            last_valid_position, collision_position, deviation_mm,
            rrt_waypoints, tcp_pos, target_position, target_orientation,
            linear,
        )

        try:
            confirm = input("  Ausführen? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if confirm in ("", "y", "yes"):
            ok = sim.move_to(target_pos, target_ori, linear=linear, speed=current_speed)
            if not ok:
                _draw_crosshair(target_position, [1, 0, 0], items,
                                f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}) FEHLER")

        tcp_items = draw_tcp()


if __name__ == "__main__":
    demo_simulation()
