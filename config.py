import math
import os

import importlib.util as _ilu

# ── Pfade ──
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
JAWS_DIR = os.path.join(PKG_DIR, "data", "meshes_jaws")
ROBOT_URDF_PATH = os.path.join(PKG_DIR, "src", "ur5e_bullet", "ur_e_description", "urdf", "ur5e.urdf")

# Waypoint-Generator: ausgelagert in das Schwestermodul waypoints.py
# (sys.path-unabhaengig geladen, wie der config-Import in __init__/sim).
_wp_spec = _ilu.spec_from_file_location("waypoints", os.path.join(PKG_DIR, "waypoints.py"))
_wp_mod = _ilu.module_from_spec(_wp_spec)
_wp_spec.loader.exec_module(_wp_mod)
parabola_waypoints = _wp_mod.parabola_waypoints

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
CAMERA_FAR_M = 0.03
CAMERA_DISPLAY_M = 0.2
# Seitlicher Versatz der beiden Kameras (Stereo-Baseline) relativ zum TCP,
# in Scanner-lokalen Koordinaten (Y-Achse), in Meter (S=1 -> 1 BU).
# Jede Kamera wird um +/- CAMERA_LATERAL_OFFSET quer zur Blickrichtung versetzt.
CAMERA_LATERAL_OFFSET = 0.0025

# ── Licht ──
LIGHT_POWER = 0.001
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

# ── Blender-Sync ──
# Legt fest, ob beim Start der Blender-Sync-Mirror mitgestartet wird
# (pusht den Roboterzustand per TCP-Socket an eine Blender-GUI-Instanz).
ENABLE_BLENDER_SYNC = True

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
PREVIEW_PAUSE = 0.6
WAYPOINT_MARKER_RADIUS = 0.001

# ── Debug: Sichtachse (Stab vom TCP) ──
# Zeichnet einen kollisionsfreien Stab vom TCP aus in Richtung der Kamera-
# Blickachse (Kandidat: TCP-lokales -Z), um die TCP->Kamera-Orientierung
# visuell zu verifizieren.
DRAW_VIEW_STICK = True
VIEW_STICK_LENGTH = 0.05
VIEW_STICK_RADIUS = 0.0015
VIEW_STICK_COLOR = [1.0, 0.3, 0.0, 1.0]

# ── Look-Target (Blickachse-Ziel) ──
# Optional zusaetzlicher Zielpunkt fuer die Blickachse (Default: Gebiss-Mittelpunkt).
# Wird als grusnes, kollisionsfreies Kuegelchen an der Waypoint-Ebenen-Hoehe dargestellt.
LOOK_TARGET_RADIUS = 0.008
LOOK_TARGET_COLOR = [0.1, 1.0, 0.3, 0.95]

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
            {"tcp_pos": [0.85, 0, 0.38], "tcp_ori_deg": [0, 0, 0], "label": "1", "use_current_seed": False},
            {"tcp_pos": [0.615, 0, 0.295], "tcp_ori_deg": [0, 90, 0], "label": "2", "use_current_seed": True},
        ],
        "jaw_pos":  [0.65, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "generator": parabola_waypoints,
        "parabola":  {"x0": 0.615, "a": 0.046, "z": 0.295, "n": 21, "y_max": 0.037, "power": 4},
        "ori_anchors": {"start": [90, 0, 0], "mid": [180, 90, 0], "end": [-90, 0, 0]},
        "look_at_jaw": True,
        "look_target": [0.661, 0.0],
    },
    "aussen2": {
        "tcp_pos":  [0.615, 0, 0.295],
        "tcp_ori_deg": [0, -90, 0],
        "approach": [
            {"tcp_pos": [0.85, 0, 0.38], "tcp_ori_deg": [0, 0, 0], "label": "1", "use_current_seed": False},
        ],
        "jaw_pos":  [0.65, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "generator": parabola_waypoints,
        "parabola":  {"x0": 0.615, "a": 0.0463, "z": 0.295, "n": 20, "y_max": 0.05, "power": 4},
        "ori_anchors": {"start": [90, 0, 0], "mid": [180, 90, 0], "end": [-90, 0, 0]},
        "look_at_jaw": True,
        "look_target": [0.65, 0.0],
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
