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
GEBISS_POSITION = _cfg_mod.GEBISS_POSITION
GEBISS_EULER = _cfg_mod.GEBISS_EULER
GEBISS_COLL_CELL = _cfg_mod.GEBISS_COLL_CELL
TOOL_OFFSET_POS = _cfg_mod.TOOL_OFFSET_POS
TOOL_OFFSET_ORN = _cfg_mod.TOOL_OFFSET_ORN
IK_LAMBDA = _cfg_mod.IK_LAMBDA
IK_TOLERANCE = _cfg_mod.IK_TOLERANCE
RRT_RESTARTS = _cfg_mod.RRT_RESTARTS
RRT_SMOOTH = _cfg_mod.RRT_SMOOTH
RRT_SEED = _cfg_mod.RRT_SEED
GHOST_COLOR = _cfg_mod.GHOST_COLOR


def _compute_orientation(look_dir):
    look = np.array(look_dir, dtype=np.float64)
    norm_val = np.linalg.norm(look)
    if norm_val < 1e-10:
        return [0.0, 0.0, 0.0]
    look = look / norm_val
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(look, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, look)
    R = np.column_stack([right, up, look])
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = [w, x, y, z]
    q = q / np.linalg.norm(q)
    return list(pybullet.getEulerFromQuaternion(q))


def _classify_normal(normal):
    z = normal[2]
    if z > 0.5:
        return "oben"
    if z < -0.5:
        return "innen"
    return "seitlich"


def _interp_points(pts, n):
    if len(pts) < 2 or n < 2:
        return list(pts)
    result = []
    total = len(pts) - 1
    for i in range(n):
        t = i / (n - 1) * total
        idx = min(int(t), total - 1)
        frac = t - idx
        p0 = np.array(pts[idx])
        p1 = np.array(pts[idx + 1])
        result.append(list(p0 + frac * (p1 - p0)))
    return result


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
            [-math.pi, -math.pi, 0, 0, -math.pi, -math.pi],
            [math.pi, math.pi, math.pi, 2*math.pi, math.pi, math.pi],
            [2*math.pi]*6,
            [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0],
        )
        self._ik_lambda = IK_LAMBDA
        self._collision_link_pairs = get_self_link_pairs(self.ur5, get_movable_joints(self.ur5))

        self.tool_offset_pos = list(TOOL_OFFSET_POS)
        self.tool_offset_orn = list(TOOL_OFFSET_ORN)
        self._last_conf = None
        self._last_target = None
        self._last_collision_tcp_pose = None
        self._last_collision_conf = None
        self._ghost_body = None
        self._last_linear_error = None
        if mirror is None:
            mirror = gui
        self._mirror = BlenderMirror(self) if mirror else None
        if self._mirror is not None:
            self._mirror.send_current()
        _color_links(self)

    def load_robot(self):
        self._current_jaw_folder = 1
        self._current_jaw_type = "lower"
        vis_path, col_path = self._resolve_jaw_paths(1, "lower")
        gebiss_vis = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=vis_path, meshScale=GEBISS_SCALE)
        gebiss_col = pybullet.createCollisionShape(
            pybullet.GEOM_MESH, fileName=col_path, meshScale=GEBISS_SCALE,
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self._gebiss = pybullet.createMultiBody(
            baseVisualShapeIndex=gebiss_vis,
            baseCollisionShapeIndex=gebiss_col,
            basePosition=GEBISS_POSITION,
            baseOrientation=pybullet.getQuaternionFromEuler(GEBISS_EULER),
        )
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

    def load_jaw(self, folder, jaw_type="lower"):
        vis_path, col_path = self._resolve_jaw_paths(folder, jaw_type)
        old_folder = self._current_jaw_folder
        old_type = self._current_jaw_type
        pybullet.setRealTimeSimulation(0)
        pybullet.removeBody(self._gebiss)
        gebiss_vis = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=vis_path, meshScale=GEBISS_SCALE)
        gebiss_col = pybullet.createCollisionShape(
            pybullet.GEOM_MESH, fileName=col_path, meshScale=GEBISS_SCALE,
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self._gebiss = pybullet.createMultiBody(
            baseVisualShapeIndex=gebiss_vis,
            baseCollisionShapeIndex=gebiss_col,
            basePosition=GEBISS_POSITION,
            baseOrientation=pybullet.getQuaternionFromEuler(GEBISS_EULER),
        )
        if self._check_jaw_collision():
            pybullet.removeBody(self._gebiss)
            vis_path2, col_path2 = self._resolve_jaw_paths(old_folder, old_type)
            gebiss_vis2 = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=vis_path2, meshScale=GEBISS_SCALE)
            gebiss_col2 = pybullet.createCollisionShape(
                pybullet.GEOM_MESH, fileName=col_path2, meshScale=GEBISS_SCALE,
                flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
            )
            self._gebiss = pybullet.createMultiBody(
                baseVisualShapeIndex=gebiss_vis2,
                baseCollisionShapeIndex=gebiss_col2,
                basePosition=GEBISS_POSITION,
                baseOrientation=pybullet.getQuaternionFromEuler(GEBISS_EULER),
            )
            pybullet.setRealTimeSimulation(1)
            print(f"  ⛔ Kollision mit Scanner – jaw abgebrochen")
            return False
        pybullet.setRealTimeSimulation(1)
        self._current_jaw_folder = folder
        self._current_jaw_type = jaw_type
        return True

    def unload_jaw(self):
        pybullet.setRealTimeSimulation(0)
        if self._gebiss is not None:
            pybullet.removeBody(self._gebiss)
            self._gebiss = None

    def load_jaw_at(self, folder, jaw_type, position, euler):
        vis_path, col_path = self._resolve_jaw_paths(folder, jaw_type)
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
        pybullet.setRealTimeSimulation(1)

    def _check_jaw_collision(self):
        for i in range(-1, self.num_joints):
            if pybullet.getClosestPoints(self._gebiss, self.ur5, 0.0, linkIndexB=i):
                return True
        return False

    def _load_jaw_verts(self):
        vis_path, _ = self._resolve_jaw_paths(self._current_jaw_folder, self._current_jaw_type)
        verts, tris = read_stl(vis_path)
        R_body = np.array(pybullet.getMatrixFromQuaternion(
            pybullet.getQuaternionFromEuler(GEBISS_EULER)
        )).reshape(3, 3)
        verts = (R_body @ (verts * np.array(GEBISS_SCALE)).T).T + np.array(GEBISS_POSITION)
        return verts, tris

    def _compute_arch_centerline(self, verts, n_points=20):
        y_vals = verts[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        if y_max - y_min < 1e-6:
            return []
        slice_hw = (y_max - y_min) / n_points * 1.5
        raw = []
        for i in range(n_points):
            y = y_min + (y_max - y_min) * i / (n_points - 1)
            mask = np.abs(y_vals - y) < slice_hw
            if mask.sum() < 3:
                continue
            xv = verts[mask, 0]
            zv = verts[mask, 2]
            raw.append((float(np.mean(xv)), float(y), float(np.mean(zv))))
        if len(raw) < 2:
            return raw
        raw.sort(key=lambda p: p[1])
        return _interp_points(raw, n_points)

    def compute_scan_path(self, path_type="outer", distance=0.08, n_points=20):
        verts, tris = self._load_jaw_verts()
        centerline = self._compute_arch_centerline(verts, n_points)
        if len(centerline) < 2:
            return []
        center = np.mean(centerline, axis=0)
        outward_normals = []
        for p in centerline:
            d = np.array(p[:2]) - np.array(center[:2])
            n = np.linalg.norm(d)
            outward_normals.append(d / n if n > 1e-6 else np.array([1.0, 0.0]))
        current_joints = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        results = []
        for i, (cp, out2d) in enumerate(zip(centerline, outward_normals)):
            out3 = np.array([out2d[0], out2d[1], 0.0])
            if path_type == "outer":
                tcp_pos = np.array(cp) + out3 * distance
                look = -out3
            elif path_type == "inner":
                tcp_pos = np.array(cp) - out3 * distance
                look = out3
            elif path_type == "top":
                tcp_pos = np.array(cp) + np.array([0, 0, distance])
                look = np.array([0, 0, -1.0])
            else:
                continue
            tcp_ori = _compute_orientation(look)
            ok = self._conf_for(
                self._tcp_to_ee(list(tcp_pos), list(tcp_ori)), seed=current_joints
            ) is not None
            results.append((list(tcp_pos), list(tcp_ori), ok))
        return results

    def _draw_scan_path(self, path, items):
        for i, (pos, ori, ok) in enumerate(path):
            color = [0, 0.8, 0] if ok else [1, 0, 0]
            h = 0.005
            items.append(pybullet.addUserDebugLine(
                [pos[0]-h, pos[1], pos[2]], [pos[0]+h, pos[1], pos[2]], color, 2))
            items.append(pybullet.addUserDebugLine(
                [pos[0], pos[1]-h, pos[2]], [pos[0], pos[1]+h, pos[2]], color, 2))
            items.append(pybullet.addUserDebugLine(
                [pos[0], pos[1], pos[2]-h], [pos[0], pos[1], pos[2]+h], color, 2))
            if i < len(path) - 1:
                npos = path[i + 1][0]
                items.append(pybullet.addUserDebugLine(pos, npos, [0, 0.6, 0], 1))
        return items

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

    def _linear_segment(self, ee_start, ee_end):
        self._last_linear_error = None
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
            prev = path[-1] if path else initial_state
            for j, v in zip(self._joint_ids, prev):
                pybullet.resetJointState(self.ur5, j, v)
            conf = self._ik((pos, quat), seed=prev)
            if conf is None:
                self._last_linear_error = ("ik", i)
                for j, v in zip(self._joint_ids, initial_state):
                    pybullet.resetJointState(self.ur5, j, v)
                return None
            if not self._joint_limits_ok(conf):
                self._last_collision_tcp_pose = (pos, quat)
                self._last_collision_conf = conf
                for idx, v in enumerate(conf):
                    if not (self._null_space[0][idx] <= v <= self._null_space[1][idx]):
                        self._last_linear_error = ("limit", i, idx)
                        break
                for j, v in zip(self._joint_ids, initial_state):
                    pybullet.resetJointState(self.ur5, j, v)
                return None
            if not self._collision_free(conf):
                self._last_collision_tcp_pose = (pos, quat)
                self._last_collision_conf = conf
                self._last_linear_error = ("collision", i)
                for j, v in zip(self._joint_ids, initial_state):
                    pybullet.resetJointState(self.ur5, j, v)
                return None
            saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
            for j, v in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, j, v)
            ls = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
            fk_err = math.sqrt(sum((ls[0][j] - pos[j])**2 for j in range(3)))
            for j, v in zip(self._joint_ids, saved):
                pybullet.resetJointState(self.ur5, j, v)
            if fk_err > 0.002:
                self._last_collision_tcp_pose = (pos, quat)
                self._last_collision_conf = path[-1] if path else initial_state
                self._last_linear_error = ("fk", i)
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

        target_configuration = self._conf_for(ee_target, seed=original_joint_positions)
        if target_configuration is None or not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
        if target_configuration is None:
            ee_start = pybullet.getLinkState(
                self.ur5, self.end_effector_index,
                computeForwardKinematics=True,
            )[:2]
            path = self._linear_segment(ee_start, ee_target)
            if path is not None:
                target_configuration = path[-1]
            else:
                saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
                for j, v in zip(self._joint_ids, self._null_space[3]):
                    pybullet.resetJointState(self.ur5, j, v)
                path = self._linear_segment(ee_start, ee_target)
                for j, v in zip(self._joint_ids, saved):
                    pybullet.resetJointState(self.ur5, j, v)
                if path is not None:
                    target_configuration = path[-1]
        if target_configuration is None:
            pybullet.configureDebugVisualizer(render_flag, 1)
            for j, v in zip(self._joint_ids, original_joint_positions):
                pybullet.resetJointState(self.ur5, j, v)
            return start_tcp, target_pos, 0.0, target_pos, None

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

    def _probe_linear_stepwise(self, target_pos, target_ori, start_tcp):
        num_steps = 100
        fk_tol = 0.0005
        seed_configuration = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        last_good_conf = list(seed_configuration)
        last_good_fraction = 0.0
        failed = False
        _, q_start_tcp = self.get_tcp_pose()
        q_target_tcp = pybullet.getQuaternionFromEuler(target_ori)

        for i in range(1, num_steps + 1):
            fraction = i / num_steps
            target_tcp_at_t = [
                start_tcp[j] + fraction * (target_pos[j] - start_tcp[j])
                for j in range(3)
            ]
            interpolated_tcp_orientation = pybullet.getQuaternionSlerp(q_start_tcp, q_target_tcp, fraction)
            ee_at_t = self._tcp_to_ee(
                target_tcp_at_t,
                list(pybullet.getEulerFromQuaternion(interpolated_tcp_orientation)),
            )
            configuration = self._ik(ee_at_t, seed=seed_configuration)
            if configuration is None:
                self._last_collision_conf = last_good_conf
                failed = True
                break
            if not self._collision_free(configuration):
                self._last_collision_conf = last_good_conf
                failed = True
                break
            saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
            for j, v in zip(self._joint_ids, configuration):
                pybullet.resetJointState(self.ur5, j, v)
            ls = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
            fk_err = math.sqrt(sum((ls[0][j] - ee_at_t[0][j])**2 for j in range(3)))
            for j, v in zip(self._joint_ids, saved):
                pybullet.resetJointState(self.ur5, j, v)
            if fk_err > fk_tol:
                self._last_collision_conf = last_good_conf
                failed = True
                break
            seed_configuration = configuration
            last_good_conf = list(configuration)
            last_good_fraction = fraction

        last_valid_position = [
            start_tcp[j] + last_good_fraction * (target_pos[j] - start_tcp[j])
            for j in range(3)
        ]
        if failed:
            last_valid_orientation = pybullet.getQuaternionSlerp(
                q_start_tcp, q_target_tcp, last_good_fraction)
            last_valid_ee = self._tcp_to_ee(
                last_valid_position,
                list(pybullet.getEulerFromQuaternion(last_valid_orientation)),
            )
            self._last_collision_tcp_pose = (last_valid_ee[0], last_valid_ee[1])
            return last_valid_position, target_tcp_at_t, 0.0, last_valid_position, None
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

        result = self._probe_linear_stepwise(target_pos, target_ori, start_tcp)
        pybullet.configureDebugVisualizer(render_flag, 1)
        return result

    def _probe_path(self, target_pos, target_ori, rrt=False):
        self._last_linear_error = None
        self._last_collision_conf = None
        if self._mirror is not None:
            self._mirror.send_current()
        if rrt:
            return self._probe_rrt(target_pos, target_ori)
        return self._probe_linear(target_pos, target_ori)

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
        if cf and hasattr(self, '_gebiss'):
            cf = not any(
                pybullet.getClosestPoints(self.ur5, self._gebiss, 0.0, linkIndexA=i)
                for i in range(-1, self.num_joints)
            )
        for i, v in zip(self._joint_ids, original):
            pybullet.resetJointState(self.ur5, i, v)
        return cf

    def _obstacles(self):
        obs = []
        if hasattr(self, '_gebiss'):
            obs.append(self._gebiss)
        return obs

    def _remove_ghost(self):
        if self._ghost_body is not None:
            pybullet.removeBody(self._ghost_body)
            self._ghost_body = None

    def _show_ghost(self):
        if self._last_collision_tcp_pose is None:
            return
        self._remove_ghost()
        conf = self._last_collision_conf
        if conf is None:
            return
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        ghost = pybullet.loadURDF(
            ROBOT_URDF_PATH, [0, 100, 0], [0, 0, 0, 1],
            flags=pybullet.URDF_USE_SELF_COLLISION,
            useFixedBase=True,
        )
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)
        for i in range(-1, pybullet.getNumJoints(ghost)):
            pybullet.changeDynamics(ghost, i, mass=0)
            pybullet.changeVisualShape(ghost, i, rgbaColor=GHOST_COLOR)
        for i, v in zip(self._joint_ids, conf):
            pybullet.resetJointState(ghost, i, v)
        pybullet.resetBasePositionAndOrientation(ghost, [0, 0, 0], [0, 0, 0, 1])
        actual_ee = pybullet.getLinkState(ghost, self.end_effector_index, computeForwardKinematics=True)
        target_pos = self._last_collision_tcp_pose[0]
        error = [target_pos[i] - actual_ee[0][i] for i in range(3)]
        pybullet.resetBasePositionAndOrientation(ghost, error, [0, 0, 0, 1])
        pybullet.setCollisionFilterGroupMask(ghost, -1, 0, 0)
        for i in range(pybullet.getNumJoints(ghost)):
            pybullet.setCollisionFilterGroupMask(ghost, i, 0, 0)
        self._ghost_body = ghost

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
        err1 = self._last_linear_error
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for j, v in zip(self._joint_ids, self._null_space[3]):
            pybullet.resetJointState(self.ur5, j, v)
        path = self._linear_segment(ee_start, ee_target)
        for j, v in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, j, v)
        pybullet.configureDebugVisualizer(render_flag, 1)
        if path is None and err1 and err1[0] == "limit":
            matches = [e for e in [err1, self._last_linear_error] if e[0] == "limit"]
            self._last_linear_error = matches[0] if matches else err1
        return path

    def move_to(self, tcp_pos, tcp_ori, linear=True, obstacles=None, tol=0.08, speed=1.0, target_conf=None):
        if obstacles is None:
            obstacles = self._obstacles()
        ee_target = self._tcp_to_ee(tcp_pos, tcp_ori)
        render_flag = pybullet.COV_ENABLE_RENDERING
        pybullet.configureDebugVisualizer(render_flag, 0)
        current_joints = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        if target_conf is not None:
            target_configuration = target_conf
        else:
            target_configuration = self._conf_for(ee_target, seed=current_joints)
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

        if linear:
            ee_start = pybullet.getLinkState(
                self.ur5, self.end_effector_index,
                computeForwardKinematics=True,
            )[:2]
            path = self._find_path(ee_start, ee_target)
            if path is not None:
                return run(path)

            err = self._last_linear_error
            if err and err[0] == "limit":
                names = ["shoulder_pan", "shoulder_lift", "elbow",
                         "wrist_1", "wrist_2", "wrist_3"]
                print(f"[nein] {names[err[2]]} am Limit – kein Pfad zu {tcp_pos}")
            else:
                print(f"[nein] Kein Pfad zu {tcp_pos}")
            pybullet.configureDebugVisualizer(render_flag, 1)
            return False

        if not self._collision_free(target_configuration):
            target_configuration = self._find_collision_free_conf(ee_target)
            if target_configuration is None:
                ee_start = pybullet.getLinkState(
                    self.ur5, self.end_effector_index,
                    computeForwardKinematics=True,
                )[:2]
                path = self._linear_segment(ee_start, ee_target)
                if path is not None:
                    target_configuration = path[-1]
                else:
                    saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
                    for j, v in zip(self._joint_ids, self._null_space[3]):
                        pybullet.resetJointState(self.ur5, j, v)
                    path = self._linear_segment(ee_start, ee_target)
                    for j, v in zip(self._joint_ids, saved):
                        pybullet.resetJointState(self.ur5, j, v)
                    if path is not None:
                        target_configuration = path[-1]
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
        if target_conf is not None:
            path[-1] = list(target_conf)
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
