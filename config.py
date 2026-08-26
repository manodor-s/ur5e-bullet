import math
import os

# ── Pfade ──
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
JAWS_DIR = os.path.join(PKG_DIR, "data", "meshes_jaws")
ROBOT_URDF_PATH = os.path.join(PKG_DIR, "src", "ur5e_bullet", "ur_e_description", "urdf", "ur5e.urdf")

# ── Gebiss ──
GEBISS_POSITION = [0.85, 0, 0.3]
GEBISS_EULER = [0, 0, math.pi / 2]
GEBISS_SCALE = [0.001, 0.001, 0.001]
GEBISS_COLL_CELL = 1.5
GEBISS_ROUGHNESS = 0.8
GEBISS_SPECULAR = 0.2

# ── Kamera (RealSense D455) ──
CAMERA_ROLL_DEG = -90
CAMERA_SENSOR_W_MM = 36.0
CAMERA_SENSOR_H_MM = CAMERA_SENSOR_W_MM * 9 / 16
CAMERA_FOV_DEG = 87
CAMERA_LENS_MM = CAMERA_SENSOR_W_MM / (2 * math.tan(math.radians(CAMERA_FOV_DEG) / 2))
CAMERA_NEAR_M = 0.001
CAMERA_FAR_M = 1.0
CAMERA_DISPLAY_M = 0.2

# ── Licht ──
LIGHT_POWER = 0.0005
LIGHT_OFFSET = [0.008, 0.025, 0.213]

# ── Render ──
RENDER_W = 1920
RENDER_H = 1080
RENDER_ENGINE = "CYCLES"
RENDER_DEVICE = "GPU"
RENDER_TRANSPARENT = False

# ── Tool-Offset (Scanner → TCP) ──
TOOL_OFFSET_POS = [0.213, 0, -0.006]
TOOL_OFFSET_ORN = [0, 0, 0, 1]
CAMERA_OFFSET = [0.008, 0, 0.213]

# ── Socket ──
SOCKET_HOST = "127.0.0.1"
SOCKET_BUFFER = 4096
SOCKET_POLL_INTERVAL = 0.05

# ── RRT ──
RRT_RESTARTS = 30
RRT_SMOOTH = 30
RRT_SEED = 0

# ── Blender Scale Factor ──
S = 1

# ── Pybullet Misc ──
IK_LAMBDA = 0.05
IK_TOLERANCE = 0.08
GHOST_COLOR = [0.2, 0.8, 1.0, 0.4]

# ── Scan-Konfiguration ──
START_POSITIONS = {
    "Aussen": {
        "tcp_pos":  [0.92, -0.05, 0.29],
        "tcp_ori":  [0.0, 0.0, math.pi / 2],
        "jaw_pos":  [0.85, 0, 0.3],
        "jaw_euler": [0, 0, math.pi / 2],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "waypoints": [
            {"tcp_pos": [0.90, -0.03, 0.29], "tcp_ori": [0.0, 0.0, math.pi / 2], "label": "1"},
            {"tcp_pos": [0.88, -0.01, 0.29], "tcp_ori": [0.0, 0.0, math.pi / 2], "label": "2"},
        ],
    },
    "Oben": {
        "tcp_pos":  [0.85, 0.0, 0.38],
        "tcp_ori":  [0.0, math.pi, math.pi / 2],
        "jaw_pos":  [0.85, 0, 0.3],
        "jaw_euler": [0, 0, math.pi / 2],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "waypoints": [],
    },
    "Innen": {
        "tcp_pos":  [0.78, -0.05, 0.29],
        "tcp_ori":  [0.0, 0.0, -math.pi / 2],
        "jaw_pos":  [0.85, 0, 0.3],
        "jaw_euler": [0, 0, math.pi / 2],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "waypoints": [],
    },
}
