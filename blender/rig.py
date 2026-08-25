import bpy
import math
import os
import json
from mathutils import Vector, Euler, Matrix, Quaternion

def _script_dir():
    candidates = []
    try:
        candidates.append(("space_data", bpy.context.space_data.text.filepath))
    except (AttributeError, TypeError):
        pass
    try:
        candidates.append(("__file__", __file__))
    except NameError:
        pass
    for txt in bpy.data.texts:
        if txt.filepath:
            candidates.append(("text", txt.filepath))
    for label, fp in candidates:
        if not fp:
            continue
        d = os.path.dirname(os.path.abspath(fp))
        if os.path.isdir(os.path.join(d, "..", "data")):
            return d
    return os.path.abspath(".")

SCRIPT_DIR = _script_dir()

ROOT = os.path.dirname(SCRIPT_DIR)
POSE_JSON = os.path.join(ROOT, "data", "urdf_data.json")
OBJ_DIR = os.path.join(ROOT, "data", "meshes")
S = 1

CAMERA_ROLL = math.radians(-90)
CAMERA_SENSOR_W = 36.0
# RealSense D455: nativer 16:9-Sensor (1280x720) -> H = W * 9/16
CAMERA_SENSOR_H = CAMERA_SENSOR_W * 9 / 16
# RealSense D455 (breiteste FOV): 87° H x 58° V
CAMERA_FOV = math.radians(87)
CAMERA_LENS_MM = CAMERA_SENSOR_W / (2 * math.tan(CAMERA_FOV / 2))
CAMERA_NEAR_M = 0.001
CAMERA_FAR_M = 0.1
CAMERA_DISPLAY_M = 0.2
# D455 nativer Aspect 16:9 (1280x720), Render in 8K
RENDER_W = 3840
RENDER_H = 2160

MESH_ROTATIONS = {
    "base_link": (math.radians(90), 0, 0),
    "shoulder_link": (0, 0, 0),
    "upper_arm_link": (math.radians(90), 0, 0),
    "forearm_link": (math.radians(90), 0, 0),
    "wrist_1_link": (math.radians(90), math.radians(90), 0),
    "wrist_2_link": (0, math.radians(180), 0),
    "wrist_3_link": (math.radians(90), 0, 0),
}

MESH_OFFSETS = {
    "base_link": (0, 0, 0),
    "shoulder_link": (0, -0.05 * S, 0),
    "upper_arm_link": (0, -0.05 * S, 0),
    "forearm_link": (0, -0.05 * S, 0),
    "wrist_1_link": (0, -0.05 * S, 0),
    "wrist_2_link": (0, -0.05 * S, 0),
    "wrist_3_link": (0, -0.05 * S, 0),
}

JOINT_AXIS_INDEX = {
    "shoulder_pan_joint": 2,
    "shoulder_lift_joint": 1,
    "elbow_joint": 1,
    "wrist_1_joint": 1,
    "wrist_2_joint": 2,
    "wrist_3_joint": 1,
}

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
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03 * S, location=(0, 0, 0))
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
            obj.scale = (S, S, S)
            obj.rotation_euler = MESH_ROTATIONS.get(link_name, (0, 0, 0))
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(rotation=True, scale=True)
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
        eb.head = Vector(pos) * S

        ax_idx = JOINT_AXIS_INDEX.get(name, 2)
        if name == "wrist_2_joint":
            eb.tail = Vector(pos) * S + Vector((0, 0, -0.05 * S))
            eb.roll = 0.0
        elif ax_idx == 2:
            eb.tail = Vector(pos) * S + Vector((0, 0, 0.05 * S))
            eb.roll = 0.0
        else:
            eb.tail = Vector(pos) * S + Vector((0, 0.05 * S, 0))
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


def _import_stl(path):
    try:
        bpy.ops.wm.stl_import(filepath=path)
    except Exception:
        bpy.ops.wm.import_mesh.stl(filepath=path)
    return bpy.context.active_object


def add_scanner_and_camera(arm_obj, meshes):
    scanner_path = os.path.join(ROOT, "src", "ur5e_bullet", "ur_e_description", "meshes", "scanner-stab.stl")
    if not os.path.exists(scanner_path):
        print("  - scanner-stab.stl nicht gefunden")
        return

    scanner_obj = _import_stl(scanner_path)
    if not scanner_obj or scanner_obj.type != "MESH":
        print("  ! Scanner-Import fehlgeschlagen")
        return

    scanner_obj.name = "scanner_stab"
    scanner_obj.scale = (S, S, S)
    bpy.ops.object.select_all(action="DESELECT")
    scanner_obj.select_set(True)
    bpy.context.view_layer.objects.active = scanner_obj
    bpy.ops.object.transform_apply(scale=True)

    scanner_obj.parent = arm_obj
    scanner_obj.parent_type = "BONE"
    scanner_obj.parent_bone = "wrist_3_joint"
    scanner_obj.matrix_parent_inverse = Matrix.Identity(4)
    # Bone-lokale Transform des scanner_link-Frames (verifiziert gegen pybullet),
    # inkl. Blender-Bone-Binding am Tail (bone length 0.05 m -> 0.05*S BU Y-Offset):
    # loc (0.002*S, 0.05*S, 0) BU, Rotation -> Quaternion (w,x,y,z)=(0,0,0.7071,0.7071)
    scanner_obj.location = Vector((0.002 * S, 0.05 * S, 0.0))
    scanner_obj.rotation_mode = "QUATERNION"
    scanner_obj.rotation_quaternion = (0.0, 0.0, 0.7071, 0.7071)
    meshes["scanner_stab"] = scanner_obj
    print("  + scanner_stab -> wrist_3_joint")

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "ScannerCamera"
    cam.data.angle = CAMERA_FOV
    cam.data.sensor_width = CAMERA_SENSOR_W
    cam.data.sensor_height = CAMERA_SENSOR_H
    cam.data.sensor_fit = "AUTO"
    cam.data.clip_start = CAMERA_NEAR_M * S
    cam.data.clip_end = CAMERA_FAR_M * S
    cam.data.display_size = CAMERA_DISPLAY_M * S
    cam.parent = scanner_obj
    cam.matrix_parent_inverse = Matrix.Identity(4)
    cam.location = Vector((0.008 * S, 0, 0.213 * S))
    cam.rotation_mode = "QUATERNION"
    base_q = Euler((0, math.radians(-90), 0), "XYZ").to_quaternion()
    cam.rotation_quaternion = base_q @ Quaternion((0, 0, 1), CAMERA_ROLL)
    print("  + ScannerCamera -> scanner_stab (Position = Pybullet-TCP, Blick +Z-Welt)")

    bpy.ops.object.light_add(type="SPOT")
    light = bpy.context.active_object
    light.name = "ScannerLight"
    light.parent = scanner_obj
    light.matrix_parent_inverse = Matrix.Identity(4)
    light.location = Vector((0.008 * S, 0, 0.213 * S))
    light.rotation_euler = (0, math.radians(-90), 0)
    light.data.energy = 3
    light.data.spot_size = math.radians(60)
    print("  + ScannerLight -> scanner_stab (Position = Pybullet-TCP, Blick +Z-Welt)")


def add_lower_jaw():
    lower_jaw_stl = os.path.join(ROOT, "data", "meshes_jaws", "1", "lower.stl")
    if not os.path.exists(lower_jaw_stl):
        print(f"  - Unterkiefer fehlt: {lower_jaw_stl}")
        return

    obj = _import_stl(lower_jaw_stl)
    if not obj or obj.type != "MESH":
        print("  ! Unterkiefer-Import fehlgeschlagen")
        return

    obj.name = "gebiss_lower"
    # STL liegt in mm vor -> 0.001*S (== 0.01 bei S=10)
    obj.scale = (0.001 * S, 0.001 * S, 0.001 * S)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    obj.location = Vector((0.85 * S, 0, 0.3 * S))
    obj.rotation_euler = (0, 0, math.radians(90))
    print(f"  + gebiss_lower -> ({obj.location.x:.2f}, {obj.location.y:.2f}, {obj.location.z:.2f})")


def setup_render():
    scene = bpy.context.scene
    scene.render.resolution_x = RENDER_W
    scene.render.resolution_y = RENDER_H
    print(f"  + Render-Aufloesung -> {RENDER_W}x{RENDER_H} (D455 16:9)")


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

    print("Fuege Scanner und Kamera hinzu...")
    add_scanner_and_camera(arm_obj, meshes)

    print("Fuege Unterkiefer hinzu...")
    add_lower_jaw()

    print("Render-Einstellungen...")
    setup_render()

    bpy.context.scene.frame_set(1)
    print("\nFertig! UR5e riggt.")
    print("Animation: blender/animate.py im Scripting-Tab oeffnen und Run Script")


if __name__ == "__main__":
    main()
