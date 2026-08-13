"""FK-Validierung: Blender-Rig vs. Pybullet.

Blender-Seite (background): setzt Pose-Bones der Armature auf definierte
Joint-Configs und exportiert die Welt-Position/-Orientierung der Link-Meshes.

    blender --background --python tests/check_fk.py -- --configs <json> --out <json>

Pybullet-Seite berechnet dieselben Configs. Vergleich in tests/compare_fk.py.
"""

import json
import os
import sys

import bpy
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
import rig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rig.SCRIPT_DIR = os.path.join(ROOT, "blender")
rig.ROOT = ROOT
rig.POSE_JSON = os.path.join(ROOT, "data", "urdf_data.json")
rig.OBJ_DIR = os.path.join(ROOT, "data", "meshes")

LINKS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

CONTROL_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

ROTATION_AXES = {
    "shoulder_pan_joint": "Y",
    "shoulder_lift_joint": "Y",
    "elbow_joint": "Y",
    "wrist_1_joint": "Y",
    "wrist_2_joint": "Y",
    "wrist_3_joint": "Y",
}


def set_bone_angle(bone, angle):
    bone.rotation_mode = "XYZ"
    axis = ROTATION_AXES[bone.name]
    idx = {"X": 0, "Y": 1, "Z": 2}[axis.lstrip("-")]
    e = [0.0, 0.0, 0.0]
    e[idx] = -angle if axis.startswith("-") else angle
    bone.rotation_euler = e


def build_scene():
    data = rig.load_urdf_data()
    meshes = rig.import_meshes(data)
    arm_obj = rig.create_armature(data)
    rig.parent_meshes(arm_obj, meshes, data)
    rig.add_scanner_and_camera(arm_obj, meshes)
    return arm_obj


def export_configs(arm_obj, configs):
    def bone_quat(name):
        bone = arm_obj.pose.bones[name]
        m = arm_obj.matrix_world @ bone.matrix
        q = m.to_quaternion()
        return [q.w, q.x, q.y, q.z]

    def qmul(a, b):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]

    def qconj(q):
        return [q[0], -q[1], -q[2], -q[3]]

    zero = {}
    for name in CONTROL_JOINTS:
        set_bone_angle(arm_obj.pose.bones[name], 0.0)
    bpy.context.view_layer.update()
    for name in LINKS:
        zero[name] = bone_quat(name)

    out = {}
    for cfg in configs:
        for name, angle in zip(CONTROL_JOINTS, cfg):
            bone = arm_obj.pose.bones.get(name)
            if bone is not None:
                set_bone_angle(bone, angle)
        bpy.context.view_layer.update()
        entry = {}
        for name in LINKS:
            bone = arm_obj.pose.bones.get(name)
            if bone is None:
                entry[name] = None
                continue
            m = arm_obj.matrix_world @ bone.matrix
            dq = qmul(bone_quat(name), qconj(zero[name]))
            entry[name] = [
                [round(v, 6) for v in m.translation],
                [round(v, 6) for v in dq],
            ]
        out[json.dumps(cfg)] = entry
    return out


def main():
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1:]
    configs_path = out_path = None
    for a in args:
        if a.startswith("--configs="):
            configs_path = a.split("=", 1)[1]
        elif a.startswith("--out="):
            out_path = a.split("=", 1)[1]
    with open(configs_path) as f:
        configs = json.load(f)
    arm_obj = build_scene()
    result = export_configs(arm_obj, configs)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[check_fk] {len(configs)} Configs -> {out_path}")


if __name__ == "__main__":
    main()
