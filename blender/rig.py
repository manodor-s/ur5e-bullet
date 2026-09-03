import bpy
import math
import os
import sys
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
sys.path.insert(0, ROOT)
from config import (
    GEBISS_SCALE,
    GEBISS_ROUGHNESS, GEBISS_SPECULAR,
    CAMERA_ROLL_DEG, CAMERA_SENSOR_W_MM, CAMERA_SENSOR_H_MM,
    CAMERA_FOV_DEG, CAMERA_LENS_MM, CAMERA_NEAR_M, CAMERA_FAR_M, CAMERA_DISPLAY_M,
    LIGHT_POWER, LIGHT_OFFSET,
    RENDER_W, RENDER_H, RENDER_ENGINE, RENDER_DEVICE, RENDER_TRANSPARENT,
    TOOL_OFFSET_POS, S,
)

CAMERA_ROLL = math.radians(CAMERA_ROLL_DEG)
CAMERA_SENSOR_W = CAMERA_SENSOR_W_MM
CAMERA_SENSOR_H = CAMERA_SENSOR_H_MM
CAMERA_FOV = math.radians(CAMERA_FOV_DEG)
POSE_JSON = os.path.join(ROOT, "data", "urdf_data.json")
OBJ_DIR = os.path.join(ROOT, "data", "meshes")

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
    cam.location = Vector((TOOL_OFFSET_POS[0] * S, TOOL_OFFSET_POS[1] * S, TOOL_OFFSET_POS[2] * S))
    cam.rotation_mode = "QUATERNION"
    base_q = Euler((0, math.radians(-90), 0), "XYZ").to_quaternion()
    cam.rotation_quaternion = base_q @ Quaternion((0, 0, 1), CAMERA_ROLL)
    bpy.context.scene.camera = cam
    print("  + ScannerCamera -> scanner_stab (Position = Pybullet-TCP, Blick +Z-Welt)")

    bpy.ops.object.light_add(type="POINT")
    light = bpy.context.active_object
    light.name = "ScannerLight"
    light.parent = scanner_obj
    light.matrix_parent_inverse = Matrix.Identity(4)
    # Licht leicht von der Kamera versetzt (LED-Ring, nicht exakt co-located)
    light.location = Vector((LIGHT_OFFSET[0] * S, LIGHT_OFFSET[1] * S, LIGHT_OFFSET[2] * S))
    light.data.energy = LIGHT_POWER
    print("  + ScannerLight -> scanner_stab (Position = Pybullet-TCP, Blick +Z-Welt)")


def _enable_backface_culling(mat):
    """Macht das Material einseitig: Rueckseite (Backfacing) wird transparent,
    Vorderseite behaelt ihr BSTF. Damit ist das Gebiss von unten/innen
    unsichtbar, von der aussen liegenden (normalen) Seite sichtbar.
    Idempotent: wird nur einmal pro Material angewendet."""
    if mat.get("backface_culling_applied"):
        return
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = GEBISS_ROUGHNESS
        bsdf.inputs["Specular IOR Level"].default_value = GEBISS_SPECULAR

    for n in list(nodes):
        if n.type == "MIX_SHADER" and n.name.startswith("BackfaceCull"):
            nodes.remove(n)
        if n.type == "BSDF_TRANSPARENT" and n.name.startswith("BackfaceCull"):
            nodes.remove(n)
        if n.type == "NEW_GEOMETRY" and n.name.startswith("BackfaceCull"):
            nodes.remove(n)

    # Vorhandene Surface-Verbindung trennen (falls vorhanden)
    mtl_out = nodes.get("Material Output")
    if mtl_out is None:
        mtl_out = nodes.new("ShaderNodeOutputMaterial")
    for l in list(links):
        if l.to_node == mtl_out and l.to_socket.name == "Surface":
            links.remove(l)

    geom = nodes.new("ShaderNodeNewGeometry")
    geom.name = "BackfaceCull_Geom"
    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.name = "BackfaceCull_Transparent"
    mix = nodes.new("ShaderNodeMixShader")
    mix.name = "BackfaceCull_Mix"
    mix.location.x = bsdf.location.x + 300
    mix.location.y = bsdf.location.y

    links.new(geom.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["BSDF"], mtl_out.inputs["Surface"])

    mat["backface_culling_applied"] = True


def remove_jaw():
    for name in ("gebiss_lower", "gebiss_upper"):
        old = bpy.data.objects.get(name)
        if old is not None:
            old.select_set(True)
            bpy.data.objects.remove(old)
    print("  - gebiss entfernt")


def replace_jaw(folder=1, jaw_type="lower", pos=None, euler_deg=None):
    remove_jaw()

    if pos is None or euler_deg is None:
        print("  ! Keine Gebisspos/Orientierung uebergeben – Gebiss nicht platziert")
        return False

    stl_path = os.path.join(ROOT, "data", "meshes_jaws", str(folder), f"{jaw_type}.stl")
    if not os.path.exists(stl_path):
        print(f"  - Jaw fehlt: {stl_path}")
        return False

    jaw = _import_stl(stl_path)
    if not jaw or jaw.type != "MESH":
        print(f"  ! Jaw-Import fehlgeschlagen: {stl_path}")
        return False

    jaw.name = f"gebiss_{jaw_type}"
    jaw.scale = (GEBISS_SCALE[0] * S, GEBISS_SCALE[1] * S, GEBISS_SCALE[2] * S)
    bpy.ops.object.select_all(action="DESELECT")
    jaw.select_set(True)
    bpy.context.view_layer.objects.active = jaw
    bpy.ops.object.transform_apply(scale=True)
    jaw.location = Vector((pos[0] * S, pos[1] * S, pos[2] * S))
    jaw.rotation_euler = (
        math.radians(euler_deg[0]),
        math.radians(euler_deg[1]),
        math.radians(euler_deg[2]),
    )

    mat = bpy.data.materials.get("GebissMaterial")
    if mat is None:
        mat = bpy.data.materials.new("GebissMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Roughness"].default_value = GEBISS_ROUGHNESS
            bsdf.inputs["Specular IOR Level"].default_value = GEBISS_SPECULAR
    _enable_backface_culling(mat)
    if jaw.data.materials:
        jaw.data.materials[0] = mat
    else:
        jaw.data.materials.append(mat)

    print(f"  + {jaw.name} -> ({jaw.location.x:.2f}, {jaw.location.y:.2f}, {jaw.location.z:.2f})")
    return True


def setup_render():
    scene = bpy.context.scene
    scene.render.resolution_x = RENDER_W
    scene.render.resolution_y = RENDER_H
    scene.render.engine = RENDER_ENGINE
    if RENDER_ENGINE == "CYCLES":
        scene.cycles.device = RENDER_DEVICE
    scene.render.film_transparent = RENDER_TRANSPARENT
    print(f"  + Render: {RENDER_W}x{RENDER_H} {RENDER_ENGINE} {RENDER_DEVICE} transparent={RENDER_TRANSPARENT}")


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

    print("Render-Einstellungen...")
    setup_render()

    bpy.context.scene.frame_set(1)
    print("\nFertig! UR5e riggt.")
    print("Animation: blender/animate.py im Scripting-Tab oeffnen und Run Script")


if __name__ == "__main__":
    main()
