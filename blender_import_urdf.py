import bpy
import math
import os
import json
from mathutils import Vector, Euler, Matrix

POSE_JSON = "/home/data/Python-Projekte/ur5e-bullet/urdf_data.json"
OBJ_DIR = "/home/data/Python-Projekte/ur5e-bullet/src/ur5e_bullet/ur_e_description/meshes/ur5e/visual"

# Per-Mesh-Rotation (Euler XYZ radians) um die OBJ-Geometrie an die
# URDF-Link-Orientierung anzupassen. Nach dem Import wird rotation applied.
MESH_ROTATIONS = {
    "base_link": (math.radians(90), 0, 0),
    "shoulder_link": (0, 0, 0),
    "upper_arm_link": (math.radians(90), 0, 0),
    "forearm_link": (math.radians(90), 0, 0),
    "wrist_1_link": (math.radians(90), math.radians(90), 0),
    "wrist_2_link": (0, math.radians(180), 0),
    "wrist_3_link": (math.radians(90), 0, 0),
}

# Location-Offset (x, y, z) in Bone-Local-Space nach dem Parenting.
MESH_OFFSETS = {
    "base_link": (0, 0, 0),
    "shoulder_link": (0, -0.05, 0),
    "upper_arm_link": (0, -0.05, 0),
    "forearm_link": (0, -0.05, 0),
    "wrist_1_link": (0, -0.05, 0),
    "wrist_2_link": (0, -0.05, 0),
    "wrist_3_link": (0, -0.05, 0),
}

# Joint axis in URDF: 0=X, 1=Y, 2=Z
JOINT_AXIS_INDEX = {
    "shoulder_pan_joint": 2,
    "shoulder_lift_joint": 1,
    "elbow_joint": 1,
    "wrist_1_joint": 1,
    "wrist_2_joint": 2,
    "wrist_3_joint": 1,
}

# Bone hierarchy (parent joint name)
JOINT_PARENT = {
    "shoulder_pan_joint": None,
    "shoulder_lift_joint": "shoulder_pan_joint",
    "elbow_joint": "shoulder_lift_joint",
    "wrist_1_joint": "elbow_joint",
    "wrist_2_joint": "wrist_1_joint",
    "wrist_3_joint": "wrist_2_joint",
}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def load_urdf_data():
    with open(POSE_JSON) as f:
        return json.load(f)


def import_meshes(data):
    meshes = {}
    link_mesh_map = data["link_mesh_map"]
    for link_name, obj_name in link_mesh_map.items():
        path = os.path.join(OBJ_DIR, obj_name)
        if not os.path.exists(path):
            print(f"  - {link_name}: Datei fehlt ({path}), erzeuge Platzhalter")
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0))
            obj = bpy.context.active_object
            obj.name = link_name
            meshes[link_name] = obj
            continue

        bpy.ops.wm.obj_import(filepath=path)
        obj = None
        for o in bpy.context.selected_objects:
            if o.type == "MESH":
                obj = o
                break
        if obj is None:
            for o in bpy.data.objects:
                if o.type == "MESH" and o.name not in meshes and o.name not in [m.name for m in meshes.values()]:
                    obj = o
                    break
        if obj:
            obj.name = link_name
            obj.rotation_euler = MESH_ROTATIONS.get(link_name, (0, 0, 0))
            if any(v != 0 for v in obj.rotation_euler):
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(rotation=True)
            meshes[link_name] = obj
            print(f"  + {link_name} -> {obj_name}")
        else:
            print(f"  ! {link_name}: Import fehlgeschlagen")
    return meshes


def create_armature(data):
    joints = data["joints"]
    arm_data = bpy.data.armatures.new("UR5e_Armature")
    arm_obj = bpy.data.objects.new("UR5e", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    ebones = arm_data.edit_bones

    for joint in joints:
        name = joint["name"]
        pos = joint["pos"]
        parent_name = joint["parent_joint"]
        axis = joint["axis"]

        eb = ebones.new(name)
        eb.head = Vector(pos)

        ax_idx = JOINT_AXIS_INDEX.get(name, 2)
        if name == "wrist_2_joint":
            eb.tail = Vector(pos) + Vector((0, 0, -0.05))
            eb.roll = 0.0
        elif ax_idx == 2:
            eb.tail = Vector(pos) + Vector((0, 0, 0.05))
            eb.roll = 0.0
        else:
            eb.tail = Vector(pos) + Vector((0, 0.05, 0))
            eb.roll = math.pi / 2

        if parent_name and parent_name in ebones:
            eb.parent = ebones[parent_name]
            eb.use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def parent_meshes(arm_obj, meshes, data):
    link_positions = data["link_positions"]
    joints = data["joints"]
    base_link = None
    if "base_link" in meshes:
        base_link = meshes["base_link"]

    link_to_joint = {j["child_link"]: j["name"] for j in joints}

    for link_name, mesh in meshes.items():
        if link_name == "base_link":
            mesh.parent = None
            mesh.matrix_world = Matrix.Identity(4)
            off = MESH_OFFSETS.get(link_name, (0, 0, 0))
            mesh.location = Vector(off)
            continue

        bone_name = link_to_joint.get(link_name)
        if not bone_name or bone_name not in arm_obj.data.bones:
            print(f"  - Kein Bone für {link_name}")
            continue

        mesh.parent = None
        mesh.matrix_world = Matrix.Identity(4)
        mesh.parent = arm_obj
        mesh.parent_type = "BONE"
        mesh.parent_bone = bone_name
        mesh.matrix_parent_inverse = Matrix.Identity(4)

        off = MESH_OFFSETS.get(link_name, (0, 0, 0))
        mesh.location = Vector(off)

        print(f"  + {link_name} -> {bone_name}")


def main():
    clear_scene()
    print("Lade URDF-Daten...")
    data = load_urdf_data()

    print("Importiere Meshes...")
    meshes = import_meshes(data)
    print(f"  {len(meshes)} Meshes")

    print("Erzeuge Armature...")
    arm_obj = create_armature(data)
    arm_obj.show_in_front = True
    print(f"  {len(arm_obj.data.bones)} Bones")

    print("Parente Meshes...")
    parent_meshes(arm_obj, meshes, data)

    bpy.context.scene.frame_set(1)
    print("\nFertig! UR5e riggt.")
    print("Animation: blender_import.py im Scripting-Tab oeffnen und Run Script")


if __name__ == "__main__":
    main()
