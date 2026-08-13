"""Regeneriert data/urdf_data.json aus dem aktuellen UR5e-URDF via Pybullet."""

import json
import os
import sys

import pybullet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "src", "ur5e_bullet", "ur_e_description", "urdf", "ur5e.urdf")
OUT = os.path.join(ROOT, "data", "urdf_data.json")

MOVABLE = {
    1: ("shoulder_pan_joint", "world_joint", "shoulder_link"),
    2: ("shoulder_lift_joint", "shoulder_pan_joint", "upper_arm_link"),
    3: ("elbow_joint", "shoulder_lift_joint", "forearm_link"),
    4: ("wrist_1_joint", "elbow_joint", "wrist_1_link"),
    5: ("wrist_2_joint", "wrist_1_joint", "wrist_2_link"),
    6: ("wrist_3_joint", "wrist_2_joint", "wrist_3_link"),
}
LINKS = {
    0: "base_link",
    1: "shoulder_link",
    2: "upper_arm_link",
    3: "forearm_link",
    4: "wrist_1_link",
    5: "wrist_2_link",
    6: "wrist_3_link",
}


def q_xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]


def main():
    pybullet.connect(pybullet.DIRECT)
    ur5 = pybullet.loadURDF(URDF, flags=pybullet.URDF_USE_SELF_COLLISION)

    joints = []
    for jidx, (name, parent, child) in MOVABLE.items():
        info = pybullet.getJointInfo(ur5, jidx)
        st = pybullet.getLinkState(ur5, jidx, computeForwardKinematics=True)
        pos, orn = st[4], st[5]  # worldLinkFramePosition/Orientation
        joints.append({
            "name": name,
            "parent_joint": parent,
            "child_link": child,
            "pos": list(pos),
            "rot": q_xyzw_to_wxyz(orn),
            "axis": list(info[13]),
        })

    link_positions = {}
    link_rotations = {}
    for lidx, name in LINKS.items():
        st = pybullet.getLinkState(ur5, lidx, computeForwardKinematics=True)
        pos, orn = st[4], st[5]
        link_positions[name] = list(pos)
        link_rotations[name] = q_xyzw_to_wxyz(orn)

    data = {
        "joints": joints,
        "link_positions": link_positions,
        "link_rotations": link_rotations,
        "link_mesh_map": {
            "base_link": "base.obj",
            "shoulder_link": "shoulder.obj",
            "upper_arm_link": "upperarm.obj",
            "forearm_link": "forearm.obj",
            "wrist_1_link": "wrist1.obj",
            "wrist_2_link": "wrist2.obj",
            "wrist_3_link": "wrist3.obj",
        },
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"[regenerate_urdf_data] -> {OUT}")


if __name__ == "__main__":
    main()
