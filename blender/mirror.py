import json
import os
import socket
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig
sys.path.insert(0, rig.ROOT)
from config import SOCKET_BUFFER, SOCKET_POLL_INTERVAL, ATTACH_VIEWPORT_TO_CAMERA

CONTROL_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_conn = None
_buffer = ""
_render_tile_count = 0
_render_remaining = 0
_render_paths = []


def send_to_host(data):
    if _conn is None:
        return
    try:
        _conn.sendall((json.dumps(data) + "\n").encode("utf-8"))
    except OSError:
        pass


def build_scene():
    rig.clear_scene()
    data = rig.load_urdf_data()
    meshes = rig.import_meshes(data)
    arm_obj = rig.create_armature(data)
    arm_obj.show_in_front = True
    rig.parent_meshes(arm_obj, meshes, data)
    rig.add_scanner_and_camera(arm_obj, meshes)
    rig.setup_render()
    bpy.context.scene.frame_set(1)
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    if ATTACH_VIEWPORT_TO_CAMERA:
                        # Viewport an scene.camera (ScannerCamera_R) heften,
                        # folgt der Roboter-Kamera-Bewegung wie ein Sucher.
                        space.region_3d.view_perspective = "CAMERA"
    print("[mirror] Szene aufgebaut")


def connect(port):
    global _conn
    for _ in range(20):
        try:
            _conn = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            break
        except OSError:
            import time
            time.sleep(0.5)
    if _conn is None:
        print(f"[mirror] Keine Verbindung zu Port {port}")
        return False
    _conn.settimeout(0.0)
    _conn.sendall(b"READY\n")
    print(f"[mirror] Verbunden mit Pybullet (Port {port})")
    return True


def _apply_joints(joints):
    arm = bpy.data.objects.get("UR5e")
    if arm is None or len(joints) < 6:
        return
    for name, angle in zip(CONTROL_JOINTS, joints):
        bone = arm.pose.bones.get(name)
        if bone is None:
            continue
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, angle, 0.0)


def _apply_tcp(tcp):
    if not tcp or len(tcp) < 3:
        return
    for name, sign in (("ScannerCamera_L", -1.0), ("ScannerCamera_R", 1.0)):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.location = (
                tcp[0] * rig.S,
                (tcp[1] + sign * rig.CAMERA_LATERAL_OFFSET) * rig.S,
                tcp[2] * rig.S,
            )
    light = bpy.data.objects.get("ScannerLight")
    if light is not None:
        light.location = (tcp[0] * rig.S, tcp[1] * rig.S, tcp[2] * rig.S)


def poll():
    global _buffer
    try:
        data = _conn.recv(SOCKET_BUFFER)
    except BlockingIOError:
        return SOCKET_POLL_INTERVAL
    if not data:
        print("[mirror] Verbindung zu Pybullet geschlossen – Mirror stoppt")
        _conn.close()
        return None
    _buffer += data.decode("utf-8", "ignore")
    while "\n" in _buffer:
        line, _buffer = _buffer.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "joints" in msg:
            _apply_joints(msg["joints"])
        if "tcp" in msg:
            _apply_tcp(msg["tcp"])
        if "render" in msg:
            global _render_tile_count, _render_remaining, _render_paths
            _render_tile_count = 0
            _render_remaining = 2
            _render_paths = []

            @bpy.app.handlers.persistent
            def _on_render_write(scene, depsgraph=None):
                global _render_tile_count
                _render_tile_count += 1
                send_to_host({"render_progress": _render_tile_count})

            @bpy.app.handlers.persistent
            def _on_render_complete(scene, depsgraph=None):
                global _render_remaining, _render_paths
                _render_remaining -= 1
                if _render_remaining <= 0:
                    send_to_host({"render_complete": list(_render_paths)})
                    for w in list(bpy.app.handlers.render_write):
                        if w.__name__ == "_on_render_write":
                            bpy.app.handlers.render_write.remove(w)
                    for c in list(bpy.app.handlers.render_complete):
                        if c.__name__ == "_on_render_complete":
                            bpy.app.handlers.render_complete.remove(c)

            bpy.app.handlers.render_write.append(_on_render_write)
            bpy.app.handlers.render_complete.append(_on_render_complete)

            def _do_render():
                global _render_paths
                _render_paths = []
                for suffix, fname in (("_L", "render_L.png"), ("_R", "render_R.png")):
                    cam = bpy.data.objects.get(f"ScannerCamera{suffix}")
                    if cam is not None:
                        bpy.context.scene.camera = cam
                    path = os.path.join(rig.ROOT, fname)
                    bpy.context.scene.render.filepath = path
                    _render_paths.append(path)
                    bpy.ops.render.render(write_still=True)
                return None
            bpy.app.timers.register(_do_render, first_interval=0)
        if "replace_jaw" in msg:
            folder = msg["replace_jaw"].get("folder", 1)
            jaw_type = msg["replace_jaw"].get("type", "lower")
            pos = msg["replace_jaw"].get("pos")
            euler = msg["replace_jaw"].get("euler")

            def _do_replace_jaw(f=folder, t=jaw_type, p=pos, e=euler):
                ok = rig.replace_jaw(f, t, pos=p, euler_deg=e)
                send_to_host({"jaw_complete": bool(ok), "jaw_log": "\n".join(rig.jaw_log)})
                return None
            bpy.app.timers.register(_do_replace_jaw, first_interval=0)
        if "jaw_unload" in msg:

            def _do_remove_jaw():
                rig.remove_jaw()
                return None
            bpy.app.timers.register(_do_remove_jaw, first_interval=0)
    return 0.05


def main():
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1:]
    port = 0
    for a in args:
        if a.startswith("--port="):
            port = int(a.split("=", 1)[1])
    if port == 0:
        print("[mirror] Kein --port angegeben")
        return
    build_scene()
    if connect(port):
        bpy.app.timers.register(poll, first_interval=SOCKET_POLL_INTERVAL)


if __name__ == "__main__":
    main()
