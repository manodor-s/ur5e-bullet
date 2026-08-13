import json
import os
import socket
import subprocess
import threading

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MIRROR_SCRIPT = os.path.join(_PKG_DIR, "..", "..", "blender", "mirror.py")


class BlenderMirror:
    """Pusht den Roboterzustand per TCP-Socket an eine Blender-GUI-Instanz."""

    def __init__(self, sim, port=0):
        self.sim = sim
        self._alive = True
        self._connected = False
        self._conn = None
        self._lock = threading.Lock()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.listen(1)
        self._sock.settimeout(1.0)
        port = self._sock.getsockname()[1]

        self._proc = self._launch_blender(port)
        if self._proc is None:
            self.close()
            return

        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _launch_blender(self, port):
        if not os.path.exists(MIRROR_SCRIPT):
            print("[mirror] blender/mirror.py nicht gefunden – Mirror deaktiviert")
            return None
        try:
            devnull = open(os.devnull, "w")
            proc = subprocess.Popen(
                ["blender", "--python", MIRROR_SCRIPT, "--", f"--port={port}"],
                stdout=devnull, stderr=devnull, close_fds=True,
            )
        except (OSError, FileNotFoundError):
            print("[mirror] Blender konnte nicht gestartet werden – Mirror deaktiviert")
            return None
        return proc

    def _accept_loop(self):
        buf = b""
        conn = None
        try:
            while self._alive:
                try:
                    conn, _ = self._sock.accept()
                    break
                except socket.timeout:
                    continue
            if conn is None:
                return
            conn.settimeout(1.0)
            self._conn = conn
            while self._alive:
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                if b"READY" in buf:
                    buf = b""
                    self._connected = True
                    self.send_current()
        except OSError:
            pass
        finally:
            self._connected = False
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass

    def send_current(self):
        try:
            joints = self.sim.get_joint_angles()
        except Exception:
            return
        self.send_message({"joints": joints})

    def send_message(self, payload):
        if not self._connected or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            pass

    def close(self):
        self._alive = False
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            if self._conn is not None:
                self._conn.close()
        except OSError:
            pass
