import json
import os
import socket
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig

CONTROL_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_conn = None
_buffer = ""


def build_scene():
    rig.clear_scene()
    data = rig.load_urdf_data()
    meshes = rig.import_meshes(data)
    arm_obj = rig.create_armature(data)
    arm_obj.show_in_front = True
    rig.parent_meshes(arm_obj, meshes, data)
    rig.add_scanner_and_camera(arm_obj, meshes)
    rig.add_lower_jaw()
    rig.setup_render()
    bpy.context.scene.frame_set(1)
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
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
    pos = (tcp[0] * rig.S, tcp[1] * rig.S, tcp[2] * rig.S)
    for name in ("ScannerCamera", "ScannerLight"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.location = pos


def poll():
    global _buffer
    try:
        data = _conn.recv(4096)
    except BlockingIOError:
        return 0.05
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
        bpy.app.timers.register(poll, first_interval=0.05)


if __name__ == "__main__":
    main()
