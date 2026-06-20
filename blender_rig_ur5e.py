"""
Erzeugt eine UR5e-Armature in Blender basierend auf URDF-Gelenkdaten.

Verwendung in Blender (Scripting > Run Script):
  1. Lade das UR5e-Modell (z.B. BlenderKit) in die Szene
  2. Führe dieses Script aus (Text Editor > Run Script)
  3. Weise im Pose Mode die Meshes den Bones zu
  4. Exportiere mit blender_import.py --armature
"""

import bpy
import math
from mathutils import Vector, Euler, Matrix


# Joint-Definitionen aus der URDF
# (name, parent_bone, origin_xyz, origin_rpy, axis)
JOINTS = [
    ("shoulder_pan_joint", None,
     (0.0, 0.0, 0.163), (0.0, 0.0, 0.0), (0, 0, 1)),
    ("shoulder_lift_joint", "shoulder_pan_joint",
     (0.0, 0.138, 0.0), (0.0, math.pi/2, 0.0), (0, 1, 0)),
    ("elbow_joint", "shoulder_lift_joint",
     (0.0, -0.131, 0.425), (0.0, 0.0, 0.0), (0, 1, 0)),
    ("wrist_1_joint", "elbow_joint",
     (0.0, 0.0, 0.392), (0.0, math.pi/2, 0.0), (0, 1, 0)),
    ("wrist_2_joint", "wrist_1_joint",
     (0.0, 0.127, 0.0), (0.0, 0.0, 0.0), (0, 0, 1)),
    ("wrist_3_joint", "wrist_2_joint",
     (0.0, 0.0, 0.1), (0.0, 0.0, 0.0), (0, 1, 0)),
]


def create_armature():
    scene = bpy.context.scene

    armature_data = bpy.data.armatures.new("UR5e_Armature")
    armature_obj = bpy.data.objects.new("UR5e", armature_data)
    scene.collection.objects.link(armature_obj)
    scene.view_layers[0].objects.active = armature_obj
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_data.edit_bones

    bone_map = {}
    parent_matrix = Matrix.Identity(4)

    for name, parent_name, xyz, rpy, axis in JOINTS:
        eb = edit_bones.new(name)
        rot = Euler(rpy).to_matrix().to_4x4()
        trans = Matrix.Translation(Vector(xyz))
        joint_matrix = parent_matrix @ trans @ rot

        eb.head = joint_matrix.to_translation()
        if axis == (0, 0, 1):
            eb.tail = eb.head + Vector((0, 0, 0.05))
            eb.roll = 0.0
        elif axis == (0, 1, 0):
            eb.tail = eb.head + Vector((0, 0.05, 0))
            eb.roll = math.pi / 2
        elif axis == (1, 0, 0):
            eb.tail = eb.head + Vector((0.05, 0, 0))
            eb.roll = 0.0

        if parent_name and parent_name in edit_bones:
            eb.parent = edit_bones[parent_name]
            eb.use_connect = False

        if parent_name:
            parent_matrix = joint_matrix
        else:
            parent_matrix = joint_matrix

        bone_map[name] = eb

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[Armature mit {len(JOINTS)} Bones erstellt: '{armature_obj.name}']")
    return armature_obj


def set_bone_axis_visuals(armature_obj):
    """Zeigt Bone-Achsen im Viewport an."""
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="POSE")
    for bone in armature_obj.pose.bones:
        bone.custom_shape = None
    bpy.ops.object.mode_set(mode="OBJECT")
    armature_obj.show_in_front = True


def print_parenting_guide():
    print("\n=== Parenting-Anleitung ===")
    print("1. Armature im Pose Mode: Armature auswählen, Pose Mode aktivieren")
    print("2. Jeweils ein Mesh + dazugehörigen Bone auswählen (Strg+Klick)")
    print("3. Strg+P > 'Bone' wählen")
    print("\nMesh → Bone Zuordnung:")
    guide = [
        ("Base", "shoulder_pan_joint"),
        ("Joint.001", "shoulder_lift_joint"),
        ("Joint.002 / Cap.001-003", "elbow_joint"),
        ("Joint.003", "wrist_1_joint"),
        ("Joint.004", "wrist_2_joint"),
        ("wrist / wrist-Bolt", "wrist_3_joint"),
        ("Gripper / Claw-* / Gripper-finger.*", "wrist_3_joint"),
    ]
    for mesh, bone in guide:
        print(f"   {mesh} → {bone}")
    print("\n4. Nach dem Parenten: Alle Meshes + Armature auswählen")
    print("5. Objekt-Modus: Strg+J (Join) um alles in ein Objekt zu packen")
    print("   ODER alles gruppiert lassen")
    print("\nDann: blender --python blender_import.py -- --json recorded_poses.json --armature")


if __name__ == "__main__":
    arm = create_armature()
    set_bone_axis_visuals(arm)
    print_parenting_guide()
