import math
import os

# ── Pfade ──
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
JAWS_DIR = os.path.join(PKG_DIR, "data", "meshes_jaws")
ROBOT_URDF_PATH = os.path.join(PKG_DIR, "src", "ur5e_bullet", "ur_e_description", "urdf", "ur5e.urdf")

# ── Gebiss ──
# Position/Orientierung kommen NICHT aus globalen Konstanten, sondern aus den
# Startpositionen (START_POSITIONS[<name>]["jaw_pos"] / ["jaw_euler_deg"]).
GEBISS_SCALE = [0.001, 0.001, 0.001]
GEBISS_COLL_CELL = 1.5
GEBISS_ROUGHNESS = 0.8
GEBISS_SPECULAR = 0.2

# ── Boot-Startposition (nur Arm) ──
# Wird beim Programmstart angefahren. Das Gebiss wird NICHT geladen –
# das passiert erst beim 'start <name>' oder 'jaw'-Befehl.
# Auf None setzen, um das Verhalten zu deaktivieren.
BOOT_START = {
    "tcp_pos":     [0.85, 0, 0.38],
    "tcp_ori_deg": [0, 0, 0],
}

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

# ── Joint Limits (Winkel in Grad) ──
# Ein Eintrag pro steuerbarem Gelenk, Reihenfolge = Joint-Reihenfolge
# (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3).
# 'rest': Neutralpose (fuer IK-Seed).
# wrist_3: Bereich auf -180..+540 (720°) vergroessert, um 180°-Orientierungsdrehungen
#          der Blickrichtung ohne Umklappen/Ans-Limit-Stossen zu ermoeglichen.
JOINTS = [
    {"name": "shoulder_pan",  "lower_deg": -180, "upper_deg": 180, "rest_deg": 0},
    {"name": "shoulder_lift", "lower_deg": -180, "upper_deg": 180, "rest_deg": -90},
    {"name": "elbow",         "lower_deg": 0,    "upper_deg": 180, "rest_deg": 90},
    {"name": "wrist_1",       "lower_deg": 0,    "upper_deg": 360, "rest_deg": 180},
    {"name": "wrist_2",       "lower_deg": -180, "upper_deg": 180, "rest_deg": -90},
    {"name": "wrist_3",       "lower_deg": -360, "upper_deg": 360, "rest_deg": 0},
]

# ── Scan-Konfiguration ──
# tcp_ori_deg / jaw_euler_deg: Orientierung in Grad
# approach (optional): Liste von Zwischen-TCP-Posen, die beim 'start <name>'
#   nacheinander angefahren werden (jeweils mit eigenem IK-Seed), bevor die
#   endgueltige tcp_pos angefahren wird – zur besseren Beeinflussung des IK.
#   Achtung: approach != waypoints (letztere sind die '+/'-navigierbaren Punkte).
START_POSITIONS = {
    "aussen1": {
        "tcp_pos":  [0.615, 0, 0.295],
        "tcp_ori_deg": [180, 90, 0],
        "approach": [
            {"tcp_pos": [0.85, 0, 0.38], "tcp_ori_deg": [0, 0, 0], "label": "1"},
            {"tcp_pos": [0.615, 0, 0.295], "tcp_ori_deg": [0, 90, 0], "label": "1"},
        ],
        "jaw_pos":  [0.65, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        # Parametrische Waypoint-Generierung: Parabel x = x0 + a*value^2, z = z.
        # 'values' = Verlaufsparameter je Punkt (hier y). Form per 'a' live anpassbar.
        # Erster Wert (0) = Scheitelpunkt x0, der dem Start-tcp_pos entspricht.
        "parabola": {"x0": 0.615, "a": 20, "z": 0.295, "n": 20},
        "values": [0.0,
                   0.0111, 0.0222, 0.0333, 0.0444, 0.0556, 0.0667, 0.0778, 0.0889, 0.1000,
                   -0.1000, -0.0889, -0.0778, -0.0667, -0.0556, -0.0444, -0.0333, -0.0222, -0.0111, -0.0056],
        # Optional: Python-Ausdruck (String) oder Callable(value) -> [rx,ry,rz] in Grad.
        # Ohne diesen Key gilt Default [0, 0, 90].
        # "ori_func": "[90, 0, 0]",
    },
    "aussen2": {
        "tcp_pos":  [0.615, 0, 0.295],
        "tcp_ori_deg": [0, -90, 0],
        "approach": [
            {"tcp_pos": [0.85, 0, 0.38], "tcp_ori_deg": [0, 0, 0], "label": "1"},
        ],
        "jaw_pos":  [0.65, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "parabola": {"x0": 0.615, "a": 18.5, "z": 0.295, "n": 20},
        "values": [0.0,
                   0.0111, 0.0222, 0.0333, 0.0444, 0.0556, 0.0667, 0.0778, 0.0889, 0.1000,
                   -0.1000, -0.0889, -0.0778, -0.0667, -0.0556, -0.0444, -0.0333, -0.0222, -0.0111, -0.0056],
    },
    "oben": {
        "tcp_pos":  [0.85, 0.0, 0.38],
        "tcp_ori_deg": [0, 180, 90],
        "approach": [],
        "jaw_pos":  [0.85, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "waypoints": [],
    },
    "innen": {
        "tcp_pos":  [0.78, -0.05, 0.29],
        "tcp_ori_deg": [0, 0, -90],
        "approach": [],
        "jaw_pos":  [0.85, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "waypoints": [],
    },
}
