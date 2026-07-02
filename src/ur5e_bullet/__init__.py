import os
import math
import json
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_URDF_PATH = os.path.join(_PKG_DIR, "ur_e_description", "urdf", "ur5e.urdf")
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")

class UR5Sim():
  
    def __init__(self, camera_attached=False, gui=True):
        pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        pybullet.setRealTimeSimulation(True)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info

        self.recording = False
        self.recorded_frames = []
        self.fps = 60
        self._quit_slider = None
        self.tool_offset_pos = None
        self.tool_offset_orn = None


    def start_recording(self):
        self.recording = True
        self.recorded_frames = []


    def stop_recording(self):
        self.recording = False


    def record_frame(self):
        if not self.recording:
            return
        joint_angles = self.get_joint_angles()
        pos, quat = self.get_tcp_pose()
        self.recorded_frames.append({
            "joint_angles": [float(a) for a in joint_angles],
            "end_effector_position": [float(p) for p in pos],
            "end_effector_orientation": [float(q) for q in quat],
        })


    def export_json(self, path="recorded_poses.json"):
        data = {
            "fps": self.fps,
            "num_frames": len(self.recorded_frames),
            "control_joints": self.control_joints,
            "frames": self.recorded_frames,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Exported {len(self.recorded_frames)} frames to {path}]")


    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        return robot
    

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)

        if self.tool_offset_pos is not None:
            ls = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
            R = np.array(pybullet.getMatrixFromQuaternion(ls[1])).reshape(3, 3)
            offset_world = R @ np.array(self.tool_offset_pos)
            position = [position[i] - offset_world[i] for i in range(3)]

        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.62))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Offset X", -0.5, 0.5, 0.22))
        self.sliders.append(pybullet.addUserDebugParameter("Offset Y", -0.5, 0.5, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Offset Z", -0.5, 0.5, 0))
        self._quit_slider = pybullet.addUserDebugParameter("SAVE & QUIT (slide right)", 0, 1, 0)


    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

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
            quat = pybullet.multiplyTransforms([0,0,0], quat, self.tool_offset_pos, self.tool_offset_orn)[1]
        return (pos, quat)

    def _move_linear(self, start, end, steps_between, sleep_each, max_depth=3):
        pos_a, ori_a = start["position"], start["orientation"]
        pos_b, ori_b = end["position"], end["orientation"]

        for t in range(1, steps_between + 1):
            frac = t / steps_between
            interp_pos = [a + frac * (b - a) for a, b in zip(pos_a, pos_b)]
            interp_ori = [a + frac * (b - a) for a, b in zip(ori_a, ori_b)]
            joint_angles = self.calculate_ik(interp_pos, interp_ori)
            self.set_joint_angles(joint_angles)
            actual_pos, _ = self.get_current_pose()
            dx = actual_pos[0] - interp_pos[0]
            dy = actual_pos[1] - interp_pos[1]
            dz = actual_pos[2] - interp_pos[2]
            err = math.sqrt(dx*dx + dy*dy + dz*dz)
            if err > 0.01 and max_depth > 0:
                mid_pos = [(a + b) / 2 for a, b in zip(pos_a, pos_b)]
                mid_ori = [(a + b) / 2 for a, b in zip(ori_a, ori_b)]
                mid = {"position": mid_pos, "orientation": mid_ori}
                self._move_linear(start, mid, steps_between, sleep_each, max_depth - 1)
                self._move_linear(mid, end, steps_between, sleep_each, max_depth - 1)
                return
            self.record_frame()
            time.sleep(sleep_each)

    def _move_joint(self, start_angles, end_angles, steps_between, sleep_each):
        for t in range(1, steps_between + 1):
            frac = t / steps_between
            interp = [a + frac * (b - a) for a, b in zip(start_angles, end_angles)]
            self.set_joint_angles(interp)
            self.record_frame()
            time.sleep(sleep_each)

    def _move_axis_aligned(self, start, end, steps_per_segment, sleep_each):
        pa = start["position"]
        pb = end["position"]
        ori = end.get("orientation", start.get("orientation", [0, 0, 0]))

        mid_xy = {"position": [pb[0], pa[1], pa[2]], "orientation": ori}
        mid_z  = {"position": [pb[0], pb[1], pa[2]], "orientation": ori}

        for segment_start, segment_end in [(start, mid_xy), (mid_xy, mid_z), (mid_z, end)]:
            if segment_start["position"] == segment_end["position"]:
                continue
            self._move_linear(segment_start, segment_end, steps_per_segment, sleep_each)

    def follow_path(self, waypoints, steps_between=50, sleep_each=0.005, dwell=1.0, home_pose=None, linear=False, axis_aligned=True):
        all_targets = []
        if home_pose is not None:
            all_targets.append(home_pose)
        all_targets.extend(waypoints)

        prev_angles = None
        prev_target = None

        for i, wp in enumerate(all_targets):
            pos = wp.get("position")
            ori = wp.get("orientation", [0, 0, 0])
            target_angles = self.calculate_ik(pos, ori)

            if i == 0:
                self.set_joint_angles(target_angles)
                self.record_frame()
                time.sleep(sleep_each)
            elif axis_aligned:
                self._move_axis_aligned(prev_target, wp, steps_between, sleep_each)
            elif linear:
                self._move_linear(prev_target, wp, steps_between, sleep_each)
            else:
                self._move_joint(prev_angles, target_angles, steps_between, sleep_each)

            prev_angles = target_angles
            prev_target = wp

            for _ in range(int(dwell * self.fps)):
                self.record_frame()
                time.sleep(1 / self.fps)

def demo_simulation(tool_offset=None):
    """ Demo program showing how to use the sim """
    sim = UR5Sim()
    if tool_offset is not None:
        sim.set_tool_offset(tool_offset)
    sim.add_gui_sliders()
    while True:
        x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
        ox = pybullet.readUserDebugParameter(sim.sliders[7])
        oy = pybullet.readUserDebugParameter(sim.sliders[8])
        oz = pybullet.readUserDebugParameter(sim.sliders[9])
        sim.set_tool_offset([ox, oy, oz])
        joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        sim.set_joint_angles(joint_angles)
        sim.check_collisions()


def record_demo_simulation(export_path=None, tool_offset=None):
    if export_path is None:
        export_path = os.path.join(os.path.dirname(_PKG_DIR), "..", "data", "poses.json")
    export_path = os.path.abspath(export_path)
    sim = None
    try:
        sim = UR5Sim()
        if tool_offset is not None:
            sim.set_tool_offset(tool_offset)
        sim.add_gui_sliders()
        sim.start_recording()

        print("[Recording: move robot with sliders. Slide 'SAVE & QUIT' to 1 or press Ctrl+C to export.]", flush=True)
        while pybullet.isConnected():
            if pybullet.readUserDebugParameter(sim._quit_slider) > 0.5:
                break
            x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
            ox = pybullet.readUserDebugParameter(sim.sliders[6])
            oy = pybullet.readUserDebugParameter(sim.sliders[7])
            oz = pybullet.readUserDebugParameter(sim.sliders[8])
            sim.set_tool_offset([ox, oy, oz])
            joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
            sim.set_joint_angles(joint_angles)
            sim.record_frame()
            time.sleep(1 / sim.fps)
    except BaseException:
        pass
    finally:
        if sim is not None and pybullet.isConnected():
            pybullet.disconnect()
        if sim is not None:
            sim.export_json(export_path)
            print(f"[Done: exported {len(sim.recorded_frames)} frames to {export_path}]", flush=True)


def run_path_simulation(waypoints, export_path=None, home_pose=None, tool_offset=None):
    if export_path is None:
        export_path = os.path.join(os.path.dirname(_PKG_DIR), "..", "data", "poses.json")
    export_path = os.path.abspath(export_path)
    sim = UR5Sim()
    if tool_offset is not None:
        sim.set_tool_offset(tool_offset)
    sim.start_recording()
    sim.follow_path(waypoints, home_pose=home_pose)
    sim.export_json(export_path)
    pybullet.disconnect()


def record_workspace(num_positions=300, steps_between=30, export_path=None, fixed_wrist=None, tool_offset=None):
    if export_path is None:
        export_path = os.path.join(os.path.dirname(_PKG_DIR), "..", "data", "workspace.json")
    export_path = os.path.abspath(export_path)
    sim = UR5Sim(gui=False)
    if tool_offset is not None:
        sim.set_tool_offset(tool_offset)

    joint_ids = [1, 2, 3, 4, 5, 6]
    joint_limits = {}
    for joint in sim.joints.values():
        if joint.controllable:
            joint_limits[joint.name] = (joint.lowerLimit, joint.upperLimit)

    sim.start_recording()
    cur_angles = sim.get_joint_angles()
    target_positions = []
    expected_frames = num_positions * steps_between
    print(f"[Start: {num_positions} Positionen x {steps_between} Schritte = {expected_frames} Frames]", flush=True)

    for i in range(num_positions):
        target = []
        for jname in sim.control_joints:
            if fixed_wrist and "wrist" in jname:
                target.append(fixed_wrist.get(jname, 0.0))
            else:
                lo, hi = joint_limits[jname]
                target.append(random.uniform(float(lo), float(hi)))

        for t in range(1, steps_between + 1):
            frac = t / steps_between
            interp = [a + frac * (b - a) for a, b in zip(cur_angles, target)]
            for idx, angle in zip(joint_ids, interp):
                pybullet.resetJointState(sim.ur5, idx, angle)
            sim.record_frame()

        for idx, angle in zip(joint_ids, target):
            pybullet.resetJointState(sim.ur5, idx, angle)
        pos, _ = sim.get_current_pose()
        target_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_positions}", flush=True)

        cur_angles = target

    sim.fps = 60
    print(f"[Frames aufgezeichnet: {len(sim.recorded_frames)}]", flush=True)
    print(f"[Exportiere nach {export_path}]", flush=True)
    data = {
        "fps": sim.fps,
        "num_frames": len(sim.recorded_frames),
        "control_joints": sim.control_joints,
        "frames": sim.recorded_frames,
        "samples": target_positions,
        "fixed_wrist": fixed_wrist,
    }
    with open(export_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Done: {len(sim.recorded_frames)} Frames, {len(target_positions)} Samples]", flush=True)
    pybullet.disconnect()


if __name__ == "__main__":
    record_demo_simulation()