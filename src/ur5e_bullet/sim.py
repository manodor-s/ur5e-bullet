import os
import math
import random
import time
import numpy as np
import pybullet

from collections import namedtuple

from pybullet_planning import plan_joint_motion
from pybullet_planning.motion_planners.smoothing import refine_waypoints
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_extend_fn
from pybullet_planning.interfaces.robots.link import get_self_link_pairs
from pybullet_planning.interfaces.robots.joint import get_movable_joints

from .blender_link import BlenderMirror
from .decimate_stl import read_stl, decimate, write_stl

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_PKG_DIR, "..", "..")
_JAWS_DIR = os.path.join(_PROJECT_ROOT, "data", "meshes_jaws")
ROBOT_URDF_PATH = os.path.join(_PKG_DIR, "ur_e_description", "urdf", "ur5e.urdf")

import importlib.util as _ilu
_cfg = _ilu.spec_from_file_location("config", os.path.join(_PROJECT_ROOT, "config.py"))
_cfg_mod = _ilu.module_from_spec(_cfg)
_cfg.loader.exec_module(_cfg_mod)
GEBISS_SCALE = _cfg_mod.GEBISS_SCALE
GEBISS_COLL_CELL = _cfg_mod.GEBISS_COLL_CELL
TOOL_OFFSET_POS = _cfg_mod.TOOL_OFFSET_POS
TOOL_OFFSET_ORN = _cfg_mod.TOOL_OFFSET_ORN
IK_LAMBDA = _cfg_mod.IK_LAMBDA
IK_TOLERANCE = _cfg_mod.IK_TOLERANCE
RRT_RESTARTS = _cfg_mod.RRT_RESTARTS
RRT_SMOOTH = _cfg_mod.RRT_SMOOTH
RRT_SEED = _cfg_mod.RRT_SEED
JOINT_LOWER_LIMITS = [math.radians(j["lower_deg"]) for j in _cfg_mod.JOINTS]
JOINT_UPPER_LIMITS = [math.radians(j["upper_deg"]) for j in _cfg_mod.JOINTS]
JOINT_REST_POSES = [math.radians(j["rest_deg"]) for j in _cfg_mod.JOINTS]


class UR5Sim():

    def __init__(self, gui=True, mirror=None):
        pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
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
            list(JOINT_LOWER_LIMITS),
            list(JOINT_UPPER_LIMITS),
            [JOINT_UPPER_LIMITS[i] - JOINT_LOWER_LIMITS[i] for i in range(6)],
            list(JOINT_REST_POSES),
        )
        self._ik_lambda = IK_LAMBDA
        self._collision_link_pairs = get_self_link_pairs(self.ur5, get_movable_joints(self.ur5))

        self.tool_offset_pos = list(TOOL_OFFSET_POS)
        self.tool_offset_orn = list(TOOL_OFFSET_ORN)
        self._last_conf = None
        self._last_target = None
        if mirror is None:
            mirror = gui
        self._mirror = BlenderMirror(self) if mirror else None
        if self._mirror is not None:
            self._mirror.send_current()
        _color_links(self)

    def load_robot(self):
        self._current_jaw_folder = 1
        self._current_jaw_type = "lower"
        self._jaw_pos = None
        self._jaw_euler = None
        self._gebiss = None
        return pybullet.loadURDF(
            ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1],
            flags=pybullet.URDF_USE_SELF_COLLISION,
        )

    @staticmethod
    def _resolve_jaw_paths(folder, jaw_type="lower"):
        stl = os.path.join(_JAWS_DIR, str(folder), f"{jaw_type}.stl")
        if not os.path.exists(stl):
            raise FileNotFoundError(f"Jaw STL nicht gefunden: {stl}")
        coll_stl = os.path.join(_JAWS_DIR, str(folder), f"{jaw_type}_coll.stl")
        if not os.path.exists(coll_stl):
            print(f"  ~ {jaw_type}_coll.stl fehlt – generiere...")
            verts, tris = read_stl(stl)
            rep, new_tris = decimate(verts, tris, GEBISS_COLL_CELL)
            write_stl(coll_stl, rep, new_tris)
            print(f"  ~ {len(new_tris)} Dreiecke (Reduktion: {len(new_tris)/len(tris):.1%})")
        return stl, coll_stl

    def load_jaw(self, folder, jaw_type="lower", position=None, euler=None):
        if position is None or euler is None:
            if self._jaw_pos is None or self._jaw_euler is None:
                print("  ! Keine Gebissposition aktiv – zuerst 'start <name>'")
                return False
            position = self._jaw_pos
            euler = self._jaw_euler
        return self.load_jaw_at(folder, jaw_type, position, euler)

    def load_jaw_at(self, folder, jaw_type, position, euler):
        vis_path, col_path = self._resolve_jaw_paths(folder, jaw_type)
        old_folder = self._current_jaw_folder
        old_type = self._current_jaw_type
        old_position = self._jaw_pos
        old_euler = self._jaw_euler
        old_body = self._gebiss
        pybullet.setRealTimeSimulation(0)
        if self._gebiss is not None:
            pybullet.removeBody(self._gebiss)
        gebiss_vis = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=vis_path, meshScale=GEBISS_SCALE)
        gebiss_col = pybullet.createCollisionShape(
            pybullet.GEOM_MESH, fileName=col_path, meshScale=GEBISS_SCALE,
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self._gebiss = pybullet.createMultiBody(
            baseVisualShapeIndex=gebiss_vis,
            baseCollisionShapeIndex=gebiss_col,
            basePosition=position,
            baseOrientation=pybullet.getQuaternionFromEuler(euler),
        )
        self._current_jaw_folder = folder
        self._current_jaw_type = jaw_type
        self._jaw_pos = position
        self._jaw_euler = euler
        if self._check_jaw_collision():
            pybullet.removeBody(self._gebiss)
            if old_body is None or old_position is None:
                self._gebiss = old_body
                pybullet.setRealTimeSimulation(1)
                print(f"  ⛔ Kollision mit Scanner – jaw abgebrochen")
                return False
            vis_path2, col_path2 = self._resolve_jaw_paths(old_folder, old_type)
            gebiss_vis2 = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=vis_path2, meshScale=GEBISS_SCALE)
            gebiss_col2 = pybullet.createCollisionShape(
                pybullet.GEOM_MESH, fileName=col_path2, meshScale=GEBISS_SCALE,
                flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            self._gebiss = pybullet.createMultiBody(
                baseVisualShapeIndex=gebiss_vis2,
                baseCollisionShapeIndex=gebiss_col2,
                basePosition=old_position,
                baseOrientation=pybullet.getQuaternionFromEuler(old_euler),
            )
            self._current_jaw_folder = old_folder
            self._current_jaw_type = old_type
            self._jaw_pos = old_position
            self._jaw_euler = old_euler
            pybullet.setRealTimeSimulation(1)
            print(f"  ⛔ Kollision mit Scanner – jaw abgebrochen")
            return False
        pybullet.setRealTimeSimulation(1)
        return True

    def unload_jaw(self):
        pybullet.setRealTimeSimulation(0)
        if self._gebiss is not None:
            pybullet.removeBody(self._gebiss)
            self._gebiss = None

    def _check_jaw_collision(self):
        for i in range(-1, self.num_joints):
            if pybullet.getClosestPoints(self._gebiss, self.ur5, 0.0, linkIndexB=i):
                return True
        return False

    def set_joint_angles(self, joint_angles):
        pybullet.setJointMotorControlArray(
            self.ur5, self._joint_ids,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*6,
            positionGains=[0.04]*6,
            forces=[self.joints[n].maxForce for n in self.control_joints],
        )
        if self._mirror is not None:
            self._mirror.send_current()

    def get_joint_angles(self):
        return [s[0] for s in pybullet.getJointStates(self.ur5, [1, 2, 3, 4, 5, 6])]

    def _ik(self, ee_pose, seed=None, max_iter=10, tol=1e-6):
        target_pos, target_quat = ee_pose
        rest = seed if seed is not None else self._null_space[3]
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, v in zip(self._joint_ids, rest):
            pybullet.resetJointState(self.ur5, i, v)
        conf = list(pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, target_pos, target_quat,
            lowerLimits=self._null_space[0], upperLimits=self._null_space[1],
            jointRanges=self._null_space[2], restPoses=rest,
            solver=pybullet.IK_DLS,
        ))
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
            for j, v in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, j, v)
            pybullet.stepSimulation()
            if self._mirror is not None:
                self._mirror.send_current()
            pos, _ = self.get_tcp_pose()
            print(f"\rX {pos[0]:.3f} Y {pos[1]:.3f} Z {pos[2]:.3f}  {_joint_deviation_line(self, short=True)}", end="", flush=True)
            if speed > 0:
                time.sleep(0.01 / speed)
        print()
        _color_links(self)

    def _probe_rrt(self, target_pos, target_ori, seed=None):
        start_tcp, _ = self.get_tcp_pose()
        original_joint_positions = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        ee_target = self._tcp_to_ee(target_pos, target_ori)
        render_flag = pybullet.COV_ENABLE_RENDERING
        pybullet.configureDebugVisualizer(render_flag, 0)

        ee_seed = seed if seed is not None else original_joint_positions
        target_configuration = self._conf_for(ee_target, seed=ee_seed)
        if target_configuration is None or not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
        if target_configuration is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            for j, v in zip(self._joint_ids, original_joint_positions):
                pybullet.resetJointState(self.ur5, j, v)
            return start_tcp, target_pos, 0.0, target_pos, None, None

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        random.seed(RRT_SEED)
        np.random.seed(RRT_SEED)
        path = plan_joint_motion(
            self.ur5, self._joint_ids, target_configuration,
            obstacles=self._obstacles(), self_collisions=True,
            restarts=RRT_RESTARTS, smooth=RRT_SMOOTH,
        )
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)

        if path is not None:
            extend_fn = get_extend_fn(self.ur5, self._joint_ids)
            path = list(refine_waypoints(path, extend_fn))
            tcp_waypoints = []
            for configuration in path:
                for j, v in zip(self._joint_ids, configuration):
                    pybullet.resetJointState(self.ur5, j, v)
                tcp, _ = self.get_tcp_pose()
                tcp_waypoints.append(list(tcp))
            for j, v in zip(self._joint_ids, original_joint_positions):
                pybullet.resetJointState(self.ur5, j, v)
            pybullet.configureDebugVisualizer(render_flag, 1)
            return target_pos, None, 0.0, target_pos, tcp_waypoints, path

        for j, v in zip(self._joint_ids, original_joint_positions):
            pybullet.resetJointState(self.ur5, j, v)
        pybullet.configureDebugVisualizer(render_flag, 1)
        return start_tcp, target_pos, 0.0, target_pos, None, None

    def _probe_path(self, target_pos, target_ori, seed=None):
        if self._mirror is not None:
            self._mirror.send_current()
        return self._probe_rrt(target_pos, target_ori, seed=seed)

    def _joint_limits_ok(self, configuration):
        for idx, v in enumerate(configuration):
            if not (self._null_space[0][idx] <= v <= self._null_space[1][idx]):
                return False
        return True

    def _collision_free(self, configuration):
        if not self._joint_limits_ok(configuration):
            return False
        original = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, v in zip(self._joint_ids, configuration):
            pybullet.resetJointState(self.ur5, i, v)
        cf = True
        for i, j in self._collision_link_pairs:
            if pybullet.getClosestPoints(self.ur5, self.ur5, 0.0, linkIndexA=i, linkIndexB=j):
                cf = False
                break
        if cf and getattr(self, '_gebiss', None) is not None:
            cf = not any(
                pybullet.getClosestPoints(self.ur5, self._gebiss, 0.0, linkIndexA=i)
                for i in range(-1, self.num_joints)
            )
        for i, v in zip(self._joint_ids, original):
            pybullet.resetJointState(self.ur5, i, v)
        return cf

    def _obstacles(self):
        obs = []
        if getattr(self, '_gebiss', None) is not None:
            obs.append(self._gebiss)
        return obs

    def move_to(self, tcp_pos, tcp_ori, obstacles=None, speed=1.0, seed=None, path=None):
        if obstacles is None:
            obstacles = self._obstacles()
        ee_target = self._tcp_to_ee(tcp_pos, tcp_ori)
        render_flag = pybullet.COV_ENABLE_RENDERING

        if path is not None:
            def run_preplanned():
                self._execute(path, speed)
                actual_tcp, actual_quat = self.get_tcp_pose()
                actual_ori = pybullet.getEulerFromQuaternion(actual_quat)
                deg = [math.degrees(a) for a in actual_ori]
                error_mm = math.sqrt(sum((actual_tcp[i] - tcp_pos[i]) ** 2 for i in range(3)))
                if error_mm > 0.005:
                    print(f"[!] ({actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f})  rx={deg[0]:.0f} ry={deg[1]:.0f} rz={deg[2]:.0f}  ({error_mm*1000:.0f} mm daneben)")
                else:
                    print(f"[ok] ({actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f})  rx={deg[0]:.0f} ry={deg[1]:.0f} rz={deg[2]:.0f}  ({error_mm*1000:.0f} mm)")
                self._save_last(tcp_pos, tcp_ori)
                return True
            return run_preplanned()

        pybullet.configureDebugVisualizer(render_flag, 0)

        current_joints = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        ik_seed = seed if seed is not None else current_joints
        target_configuration = self._conf_for(ee_target, seed=ik_seed)
        if target_configuration is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            print(f"[nein] Ziel {tcp_pos} nicht erreichbar")
            return False

        def run(path, label=""):
            pybullet.configureDebugVisualizer(render_flag, 1)
            self._execute(path, speed)
            actual_tcp, actual_quat = self.get_tcp_pose()
            actual_ori = pybullet.getEulerFromQuaternion(actual_quat)
            deg = [math.degrees(a) for a in actual_ori]
            error_mm = math.sqrt(sum((actual_tcp[i]-tcp_pos[i])**2 for i in range(3)))
            if error_mm > 0.005:
                print(f"[!] ({actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f})  rx={deg[0]:.0f} ry={deg[1]:.0f} rz={deg[2]:.0f}  ({error_mm*1000:.0f} mm daneben){label}")
            else:
                print(f"[ok] ({actual_tcp[0]:.3f}, {actual_tcp[1]:.3f}, {actual_tcp[2]:.3f})  rx={deg[0]:.0f} ry={deg[1]:.0f} rz={deg[2]:.0f}  ({error_mm*1000:.0f} mm){label}")
            self._save_last(tcp_pos, tcp_ori)
            return True

        if not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
        if target_configuration is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            print(f"[nein] Ziel {tcp_pos} kollidiert")
            return False
        random.seed(RRT_SEED)
        np.random.seed(RRT_SEED)
        path = plan_joint_motion(
            self.ur5, self._joint_ids, target_configuration,
            obstacles=obstacles, self_collisions=True,
            restarts=RRT_RESTARTS, smooth=RRT_SMOOTH,
        )
        if path is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            print(f"[nein] Kein Pfad zu {tcp_pos}")
            return False
        extend_fn = get_extend_fn(self.ur5, self._joint_ids)
        path = list(refine_waypoints(path, extend_fn))
        return run(path)

    def _save_last(self, tcp_pos, tcp_ori):
        self._last_conf = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        self._last_target = (tcp_pos, tcp_ori)
        if self._mirror is not None:
            self._mirror.send_current()

    def get_current_pose(self):
        ls = pybullet.getLinkState(
            self.ur5, self.end_effector_index, computeForwardKinematics=True
        )
        return ls[0], ls[1]

    def set_tool_offset(self, pos, ori=(0, 0, 0, 1)):
        self.tool_offset_pos = list(pos)
        self.tool_offset_orn = list(ori)
        if self._mirror is not None:
            self._mirror.send_current()

    def clear_tool_offset(self):
        self.tool_offset_pos = None
        self.tool_offset_orn = None
        if self._mirror is not None:
            self._mirror.send_current()

    def get_tcp_pose(self):
        pos, quat = self.get_current_pose()
        if self.tool_offset_pos is not None:
            R = np.array(pybullet.getMatrixFromQuaternion(quat)).reshape(3, 3)
            pos = [pos[i] + (R @ np.array(self.tool_offset_pos))[i] for i in range(3)]
            quat = pybullet.multiplyTransforms(
                [0, 0, 0], quat, self.tool_offset_pos, self.tool_offset_orn
            )[1]
        return pos, quat

    def get_tcp_in_scanner_frame(self):
        tcp_world, _ = self.get_tcp_pose()
        scanner = self.joints["scanner_joint"].id
        sc = pybullet.getLinkState(self.ur5, scanner, computeForwardKinematics=True)
        R = np.array(pybullet.getMatrixFromQuaternion(sc[5])).reshape(3, 3)
        local = R.T @ (np.array(tcp_world) - np.array(sc[4]))
        return [float(v) for v in local]


def _color_links(sim):
    for i, jid in enumerate(sim._joint_ids):
        angle = pybullet.getJointState(sim.ur5, jid)[0]
        low = sim._null_space[0][i]
        high = sim._null_space[1][i]
        mid = (low + high) / 2
        max_dev = (high - low) / 2
        ratio = abs(angle - mid) / max_dev if max_dev else 0
        ratio = min(1.0, ratio)
        pybullet.changeVisualShape(sim.ur5, jid, rgbaColor=[ratio, 1 - ratio, 0, 1])


def _joint_deviation_line(sim, short=False):
    names = ["pa", "li", "el", "w1", "w2", "w3"] if short else ["pan", "lift", "elbow", "w1", "w2", "w3"]
    parts = []
    for i, jid in enumerate(sim._joint_ids):
        angle = pybullet.getJointState(sim.ur5, jid)[0]
        deg = math.degrees(angle)
        low = math.degrees(sim._null_space[0][i])
        high = math.degrees(sim._null_space[1][i])
        mid = (low + high) / 2
        deviation = deg - mid
        max_dev = (high - low) / 2
        ratio = abs(deviation) / max_dev if max_dev else 0
        r = min(255, int(ratio * 255))
        g = min(255, int((1 - ratio) * 255))
        parts.append(f"\033[38;2;{r};{g};0m{names[i]}{deg:+6.0f}°\033[0m")
    if short:
        return " ".join(parts)
    return "Gelenke: " + " ".join(parts)
