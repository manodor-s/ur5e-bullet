import json
import os
import sys
import bpy
from mathutils import Vector, Quaternion, Euler


def load_poses(json_path):
    with open(json_path) as f:
        return json.load(f)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_end_effector_track(data):
    fps = data["fps"]
    frames = data["frames"]
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = len(frames)

    bpy.ops.object.empty_add(type="SPHERE", location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = "EndEffector"

    for i, f in enumerate(frames):
        frame = i + 1
        pos = f["end_effector_position"]
        quat = f["end_effector_orientation"]

        scene.frame_set(frame)
        empty.location = Vector(pos)
        empty.rotation_mode = "QUATERNION"
        empty.rotation_quaternion = Quaternion([quat[3], quat[0], quat[1], quat[2]])

        empty.keyframe_insert(data_path="location", index=-1)
        empty.keyframe_insert(data_path="rotation_quaternion", index=-1)

    scene.frame_set(1)
    print(f"[EndEffector: {len(frames)} Frames auf '{empty.name}' ]")


JOINT_AXIS = {
    "shoulder_pan_joint": 1,
    "shoulder_lift_joint": 1,
    "elbow_joint": 1,
    "wrist_1_joint": 1,
    "wrist_2_joint": 1,
    "wrist_3_joint": 1,
}


def _set_rotation(obj_or_bone, angle, axis):
    euler = [0.0, 0.0, 0.0]
    euler[axis] = angle
    obj_or_bone.rotation_euler = Euler(euler, "XYZ")


def keyframe_linkforge_joints(data):
    control_joints = data["control_joints"]
    frames = data["frames"]
    scene = bpy.context.scene

    found = []
    for name in control_joints:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "EMPTY":
            found.append((obj, JOINT_AXIS.get(name, 2)))
            print(f"  -> Joint gefunden: '{name}'")
        else:
            print(f"  !! Joint nicht gefunden: '{name}'")

    if not found:
        print("[Fehler: Keine Joint-Empties gefunden]")
        return

    for i, f in enumerate(frames):
        frame = i + 1
        angles = f["joint_angles"]
        scene.frame_set(frame)
        for (obj, axis), angle in zip(found, angles):
            _set_rotation(obj, angle, axis)
            obj.keyframe_insert(data_path="rotation_euler", index=-1)

    scene.frame_set(1)
    print(f"[Joint-Keyframes: {len(frames)} Frames auf {len(found)} Gelenke]")


def keyframe_armature_joints(data):
    control_joints = data["control_joints"]
    frames = data["frames"]
    scene = bpy.context.scene

    armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature = obj
            break

    if not armature:
        print("[Fehler: Keine Armature in der Szene gefunden]")
        return

    pose_bones = armature.pose.bones
    found = []
    for name in control_joints:
        if name in pose_bones:
            found.append((pose_bones[name], JOINT_AXIS.get(name, 2)))
            print(f"  -> Bone gefunden: '{name}'")
        else:
            print(f"  !! Bone nicht gefunden: '{name}'")

    if not found:
        print("[Fehler: Keine passenden Pose-Bones gefunden]")
        return

    for i, f in enumerate(frames):
        frame = i + 1
        angles = f["joint_angles"]
        scene.frame_set(frame)
        for (bone, axis), angle in zip(found, angles):
            bone.rotation_mode = "XYZ"
            _set_rotation(bone, angle, axis)
            bone.keyframe_insert(data_path="rotation_euler", index=-1)

    scene.frame_set(1)
    print(f"[Armature-Keyframes: {len(frames)} Frames auf {len(found)} Bones]")


if __name__ == "__main__":
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(bpy.context.space_data.text.filepath))
    ROOT = os.path.dirname(SCRIPT_DIR)

    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
    else:
        args = []

    path = None
    mode = "empty"

    for a in args:
        if a.startswith("--json="):
            path = a.split("=", 1)[1]
        elif a == "--linkforge":
            mode = "linkforge"
        elif a == "--armature":
            mode = "armature"

    if mode == "empty" and any(o.type == "ARMATURE" for o in bpy.data.objects):
        mode = "armature"
        print("[Auto-Modus: Armature gefunden -> armature]")

    if path is None and len(args) >= 1 and not args[0].startswith("--"):
        path = args[0]

    if path is None:
        candidates = [
            os.path.join(ROOT, "data", "poses.json"),
            os.path.join(ROOT, "recorded_poses.json"),
            bpy.path.abspath("//poses.json"),
            bpy.path.abspath("//recorded_poses.json"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), "poses.json")

    if not os.path.exists(path):
        print(f"[Fehler: Datei nicht gefunden: {path}]")
        print(f"[Pfad anpassen: --json=<absoluter-pfad-zu>/poses.json]")
    else:
        data = load_poses(path)
        if mode == "empty":
            clear_scene()
            add_end_effector_track(data)
        elif mode == "linkforge":
            keyframe_linkforge_joints(data)
            add_end_effector_track(data)
        elif mode == "armature":
            keyframe_armature_joints(data)
            add_end_effector_track(data)
