import os
import math
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

    def _conf_for(self, ee_pose):
        conf = self._ik(ee_pose)
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

    def _execute(self, path, speed=1.0):
        for conf in path:
            for i, val in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, i, val)
            pybullet.stepSimulation()
            if speed > 0:
                time.sleep(0.01 / speed)

    def _linear_segment(self, ee_start, ee_end, seed=None):
        if seed is None:
            seed = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        steps = 100
        path = []
        cur = seed
        for i in range(1, steps + 1):
            t = i / steps
            pos = [
                ee_start[0][j] + t * (ee_end[0][j] - ee_start[0][j])
                for j in range(3)
            ]
            quat = pybullet.getQuaternionSlerp(ee_start[1], ee_end[1], t)
            conf = self._ik((pos, quat), seed=cur)
            if conf is None or not self._collision_free(conf):
                return None
            path.append(conf)
            cur = conf
        return path

    def _probe_path(self, target_pos, target_ori):
        """Prüft den direkten linearen Pfad.
        Rückgabe: (last_good, fail_pos, dev_mm, actual_end)
          last_good:   letzte gültige Position auf dem Pfad (TCP)
          fail_pos:    Position an der es scheitert (None = OK)
          dev_mm:      Abweichung am Ziel in mm (0 = perfekt)
          actual_end:  tatsächlich erreichter TCP (bei dev>0)
        """
        ee_target = self._tcp_to_ee(target_pos, target_ori)
        ee_start = pybullet.getLinkState(
            self.ur5, self.end_effector_index,
            computeForwardKinematics=True,
        )[:2]
        start_tcp, _ = self.get_tcp_pose()
        path = self._linear_segment(ee_start, ee_target)
        if path is not None:
            steps = len(path)
            saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
            last_good_t = 0.0
            for i, conf in enumerate(path):
                t = (i + 1) / steps
                tcp_target = [
                    start_tcp[j] + t * (target_pos[j] - start_tcp[j])
                    for j in range(3)
                ]
                for j, v in zip(self._joint_ids, conf):
                    pybullet.resetJointState(self.ur5, j, v)
                actual, _ = self.get_tcp_pose()
                dev = math.sqrt(sum((actual[j]-tcp_target[j])**2 for j in range(3))) * 1000
                if dev <= 5:
                    last_good_t = t
                else:
                    break
            # letztes gutes Ziel auf der Geraden
            last_good_pos = [
                start_tcp[j] + last_good_t * (target_pos[j] - start_tcp[j])
                for j in range(3)
            ]
            # Abweichung am Pfadende
            conf = path[-1]
            for j, v in zip(self._joint_ids, conf):
                pybullet.resetJointState(self.ur5, j, v)
            actual_end, _ = self.get_tcp_pose()
            end_dev = math.sqrt(sum((actual_end[j]-target_pos[j])**2 for j in range(3))) * 1000
            for j, v in zip(self._joint_ids, saved):
                pybullet.resetJointState(self.ur5, j, v)
            if last_good_t >= 1.0 and end_dev <= 5:
                return target_pos, None, 0.0, actual_end
            return last_good_pos, None, end_dev, actual_end
        steps = 100
        cur = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        last = ee_start[0]
        for i in range(1, steps + 1):
            t = i / steps
            pos = [
                ee_start[0][j] + t * (ee_target[0][j] - ee_start[0][j])
                for j in range(3)
            ]
            quat = pybullet.getQuaternionSlerp(ee_start[1], ee_target[1], t)
            conf = self._ik((pos, quat), seed=cur)
            if conf is None or not self._collision_free(conf):
                return last, pos, 0.0, last
            cur = conf
            last = pos
        return last, None, 0.0, last

    def _collision_free(self, conf):
        saved = [s[0] for s in pybullet.getJointStates(self.ur5, self._joint_ids)]
        for i, v in zip(self._joint_ids, conf):
            pybullet.resetJointState(self.ur5, i, v)
        ok = True
        # Self-collision: adjacent links may touch
        for i in range(-1, self.num_joints):
            for j in range(i+1, self.num_joints):
                if abs(i - j) <= 1:
                    continue
                pts = pybullet.getClosestPoints(
                    self.ur5, self.ur5, 0.0,
                    linkIndexA=i, linkIndexB=j,
                )
                if pts:
                    ok = False
                    break
            if not ok:
                break
        # Table collision
        if ok and hasattr(self, '_table'):
            for i in range(-1, self.num_joints):
                pts = pybullet.getClosestPoints(
                    self.ur5, self._table, 0.0,
                    linkIndexA=i,
                )
                if pts:
                    ok = False
                    break
        for i, v in zip(self._joint_ids, saved):
            pybullet.resetJointState(self.ur5, i, v)
        return ok

    def _via_point(self, start_pos, target_pos, lift=0.1):
        z = max(start_pos[2], target_pos[2]) + lift
        return [(start_pos[0], start_pos[1], z),
                (target_pos[0], target_pos[1], z)]

    def _find_path(self, ee_start, ee_target):
        """Direct linear path → multiple via‑strategien → None."""
        for seed in [None, list(self._null_space[3])]:
            path = self._linear_segment(ee_start, ee_target, seed=seed)
            if path is not None:
                return path

        s_pos, s_quat = ee_start
        t_pos, t_quat = ee_target
        max_z = max(s_pos[2], t_pos[2])
        ori_flat = pybullet.getQuaternionFromEuler([0, 0, 0])

        for seed in [None, list(self._null_space[3])]:
            strategies = []
            for lift in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
                z = max_z + lift
                strategies.append([
                    ([s_pos[0], s_pos[1], z], ori_flat),
                    ([t_pos[0], t_pos[1], z], ori_flat),
                ])
            for z in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for x in [0.6, 0.75, 0.85]:
                    strategies.append([
                        ([x, 0, z], ori_flat),
                        ([x, 0, z], ori_flat),
                    ])
            for z in [0.3, 0.4, 0.5, 0.6, 0.7]:
                strategies.append([
                    ([s_pos[0], s_pos[1], z], ori_flat),
                    ([0.6, 0, z], ori_flat),
                    ([t_pos[0], t_pos[1], z], ori_flat),
                ])

            for via_poses in strategies:
                full = []
                prev = ee_start
                cur_seed = seed
                ok = True
                for vp in via_poses:
                    seg = self._linear_segment(prev, vp, seed=cur_seed)
                    if seg is None:
                        ok = False
                        break
                    full.extend(seg)
                    prev = vp
                    cur_seed = seg[-1]
                if ok:
                    seg = self._linear_segment(prev, ee_target, seed=cur_seed)
                    if seg is not None:
                        full.extend(seg)
                        return full
        return None

    def move_to(self, tcp_pos, tcp_ori, linear=True, obstacles=[], tol=0.08, speed=1.0):
        ee_target = self._tcp_to_ee(tcp_pos, tcp_ori)
        target_conf = self._conf_for(ee_target)
        if target_conf is None:
            print(f"[nein] Ziel {tcp_pos} nicht erreichbar")
            return False

        def run(path, label=""):
            self._execute(path, speed)
            actual, _ = self.get_tcp_pose()
            dx = math.sqrt(sum((actual[i]-tcp_pos[i])**2 for i in range(3)))
            if dx > 0.005:
                print(f"[!] {tcp_pos} → {dx*1000:.0f} mm daneben")
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

            # RRT‑Fallback nur wenn die Zielkonfiguration kollisionsfrei ist
            if self._collision_free(target_conf):
                print(f"  → versuche RRT")
                path = plan_joint_motion(
                    self.ur5, self._joint_ids, target_conf,
                    obstacles=obstacles, self_collisions=True,
                )
                if path is not None:
                    return run(path, " (RRT)")

            print(f"[nein] Kein Pfad zu {tcp_pos}")
            return False

        if not self._collision_free(target_conf):
            print(f"[nein] Ziel {tcp_pos} kollidiert")
            return False
        path = plan_joint_motion(
            self.ur5, self._joint_ids, target_conf,
            obstacles=obstacles, self_collisions=True,
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


def demo_simulation():
    sim = UR5Sim()

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

    print("── UR5e Demo ──────────────────────────────")
    print("Format: x y z rx ry rz oder 'q' zum Beenden")
    print("  Tool-Offset: 'o x y z' (z. B. o 0 0 0.15)")
    print("  RRT-Modus:   'r x y z rx ry rz'")
    print("  Geschw.  0.2: 's 0.2 x y z rx ry rz'")
    print("  Reset:       '@'  (nach manuellem Ziehen)")
    print("────────────────────────────────────────────")
    items = []
    tcp_items = draw_tcp()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split()
        if parts[0] == "q":
            break
        if parts[0] == "o" and len(parts) == 4:
            sim.set_tool_offset([float(parts[1]), float(parts[2]), float(parts[3])])
            continue
        if parts[0] == "@" and sim._last_conf is not None:
            sim._execute([sim._last_conf])
            tcp_items = draw_tcp()
            continue

        for item in items + tcp_items:
            pybullet.removeUserDebugItem(item)
        items.clear()
        tcp_items = []

        tcp_pos, _ = sim.get_tcp_pose()

        speed = 1.0
        linear = True
        idx = 0
        if parts[0] == "s":
            speed = float(parts[1])
            idx = 2
            if speed <= 0:
                speed = 1.0
        if parts[0] == "r":
            linear = False
            idx = 1
        if parts[0] == "s" and parts[idx] == "r":
            linear = False
            idx += 1
        vals = [float(v) for v in parts[idx:]]
        if len(vals) < 3:
            tcp_items = draw_tcp()
            if sim._last_target:
                dtcp = math.sqrt(sum((tcp_pos[i]-sim._last_target[0][i])**2 for i in range(3)))
                if dtcp > 0.01:
                    print(f"  ⚠ {dtcp*1000:.0f}mm manuell verschoben (@ zum Reset)")
            continue
        pos = vals[:3]
        ori = vals[3:6] if len(vals) >= 6 else [0, 0, 0]

        tcp_pos, _ = sim.get_tcp_pose()
        _draw_crosshair(pos, [1, 1, 0], items,
                        f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        last, fail, dev, actual_end = sim._probe_path(pos, ori)
        if fail is None and dev == 0.0:
            items.append(pybullet.addUserDebugLine(
                tcp_pos, pos, [0, 1, 0], 2,
            ))
            target_pos = pos
            target_ori = ori
        elif fail is None and dev > 0:
            items.append(pybullet.addUserDebugLine(
                tcp_pos, last, [0, 1, 0], 2,
            ))
            items.append(pybullet.addUserDebugLine(
                last, pos, [1, 0, 0], 2,
            ))
            ab_len = math.sqrt(sum((pos[i]-tcp_pos[i])**2 for i in range(3)))
            t = math.sqrt(sum((last[i]-tcp_pos[i])**2 for i in range(3))) / ab_len if ab_len > 0 else 0
            _, cur_quat = sim.get_tcp_pose()
            mid_quat = pybullet.getQuaternionSlerp(
                cur_quat, pybullet.getQuaternionFromEuler(ori), t,
            )
            target_pos = last
            target_ori = list(pybullet.getEulerFromQuaternion(mid_quat))
            print(f"  ⚠ {dev:.0f}mm Abweichung – fahre zu ({last[0]:.3f}, {last[1]:.3f}, {last[2]:.3f})")
        else:
            items.append(pybullet.addUserDebugLine(
                tcp_pos, last, [0, 1, 0], 2,
            ))
            items.append(pybullet.addUserDebugLine(
                last, pos, [1, 0, 0], 2,
            ))
            _draw_crosshair(fail, [1, 0, 0], items)
            target_pos = pos
            target_ori = ori
            print(f"  ⛔ Kollision bei ({fail[0]:.3f}, {fail[1]:.3f}, {fail[2]:.3f})")

        try:
            confirm = input("  Ausführen? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if confirm in ("", "y", "yes"):
            ok = sim.move_to(target_pos, target_ori, linear=linear, speed=speed)
            if not ok:
                _draw_crosshair(pos, [1, 0, 0], items,
                                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) FEHLER")

        tcp_items = draw_tcp()


if __name__ == "__main__":
    demo_simulation()
