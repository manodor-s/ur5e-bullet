import bpy
import os
import json
from mathutils import Vector

JOINT_LIMITS = {
    "shoulder_pan_joint": (-6.28318530718, 6.28318530718),
    "shoulder_lift_joint": (-6.28318530718, 6.28318530718),
    "elbow_joint": (-3.14159265359, 3.14159265359),
    "wrist_1_joint": (-6.28318530718, 6.28318530718),
    "wrist_2_joint": (-6.28318530718, 6.28318530718),
    "wrist_3_joint": (-6.28318530718, 6.28318530718),
}


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
WORKSPACE_JSON = os.path.join(ROOT, "data", "workspace.json")


def load_workspace_data():
    with open(WORKSPACE_JSON) as f:
        return json.load(f)


def create_point_cloud(positions, name="Workspace", hull=False):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    verts = [Vector(p) for p in positions]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    if hull and len(verts) >= 4:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.convex_hull()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        print(f"  Hülle: {len(mesh.vertices)} Vertices, {len(mesh.polygons)} Faces")

    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Emission Strength"].default_value = 1.0
        bsdf.inputs["Emission Color"].default_value = (0.0, 0.6, 1.0, 1.0)
        bsdf.inputs["Alpha"].default_value = 0.3
    mat.blend_method = "BLEND"
    mesh.materials.append(mat)

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.shading.type = "MATERIAL"
            break

    return obj


def main(num_samples=20000, hull=True, fixed_wrist=None):
    from_render = os.path.exists(WORKSPACE_JSON)
    if from_render:
        print(f"Lade pybullet-Daten aus {WORKSPACE_JSON}...")
        data = load_workspace_data()
        positions = data["samples"]
        print(f"  {len(positions)} Samples geladen")
    else:
        print("[pybullet-Daten nicht gefunden. Bitte zuerst record_workspace() ausführen.]")
        print("  python -c 'from ur5e_bullet import record_workspace; record_workspace()'")
        return

    name = "Workspace"
    if fixed_wrist:
        label = "_".join(f"{n}={a}" for n, a in fixed_wrist.items())
        name = f"Workspace_wrist_{label}"

    obj = create_point_cloud(positions, name, hull=hull)

    bpy.context.view_layer.objects.active = obj
    print(f"\nFertig! '{obj.name}' | {len(positions)} Punkte | {'Hülle' if hull else 'Wolke'}")
    print("Viewport auf Material Preview (Z) schalten für Transparenz.")
    print("Animation: animate.py laden und Run Script – zeigt Roboter-Bewegung durch den Workspace.")


if __name__ == "__main__":
    main(num_samples=20000, hull=False)
