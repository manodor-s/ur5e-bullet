"""Vergleicht Blender-FK-Export (tests/check_fk.py) mit Pybullet-Link-States."""

import json
import math
import sys

import pybullet

URDF = "src/ur5e_bullet/ur_e_description/urdf/ur5e.urdf"

LINKS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
LINK_INDICES = [1, 2, 3, 4, 5, 6]
CONTROL_JOINTS = [1, 2, 3, 4, 5, 6]
S = 10.0


def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]


def qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def main():
    blender_file = sys.argv[1]
    with open(blender_file) as f:
        blender_data = json.load(f)

    pybullet.connect(pybullet.DIRECT)
    ur5 = pybullet.loadURDF(URDF, flags=pybullet.URDF_USE_SELF_COLLISION)

    for j in range(6):
        pybullet.resetJointState(ur5, j + 1, 0)
    zero = {}
    for idx in LINK_INDICES:
        o = pybullet.getLinkState(ur5, idx, computeForwardKinematics=True)[5]
        zero[idx] = [o[3], o[0], o[1], o[2]]

    stats = {link: {"pos": 0.0, "ori": 0.0, "worst": None} for link in LINKS}
    for cfg_key, b_entry in blender_data.items():
        cfg = json.loads(cfg_key)
        for j, v in zip(CONTROL_JOINTS, cfg):
            pybullet.resetJointState(ur5, j, v)
        for link, idx in zip(LINKS, LINK_INDICES):
            if b_entry[link] is None:
                continue
            bpos, bdelta = b_entry[link]
            st = pybullet.getLinkState(ur5, idx, computeForwardKinematics=True)
            pos, quat_xyzw = st[4], st[5]  # worldLinkFramePosition/Orientation
            pq = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            pdelta = qmul(pq, qconj(zero[idx]))
            bpos_m = [v / S for v in bpos]
            dpos = max(abs(a - b) for a, b in zip(pos, bpos_m)) * 1000.0  # mm
            dot = abs(sum(a * b for a, b in zip(bdelta, pdelta)))
            dori = 2 * math.acos(min(1.0, dot)) * 180.0 / math.pi  # deg
            if dpos > stats[link]["pos"] or dori > stats[link]["ori"]:
                stats[link]["pos"] = max(stats[link]["pos"], dpos)
                stats[link]["ori"] = max(stats[link]["ori"], dori)
                stats[link]["worst"] = json.dumps(cfg)

    print(f"{'Link':<16}{'max mm':>10}{'max deg':>10}   worst config")
    for link in LINKS:
        s = stats[link]
        print(f"{link:<16}{s['pos']:>10.2f}{s['ori']:>10.2f}   {s['worst']}")


if __name__ == "__main__":
    main()
