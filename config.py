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

# ── Waypoint-Generator (Parabel) ──
# Echte Python-Funktion, kann frei angepasst werden. Sie wird je Startposition
# ueber den Key 'generator' referenziert und berechnet Position UND Orientierung.
#
# Position (Parabel):   x = x0 + a*value^2,  y = value,  z = z
#   value-laeuft kleinteilig von -y_max .. +y_max (verteilt auf n Punkte).
#
# Orientierung (3 Achsen getrennt anpassbar):
#   rot[k] = ori0[k] + ori_scale[k] * frac
#   - ori0    : Startwinkel je Achse (Grad); Fallback: cfg['tcp_ori_deg']
#   - ori_scale: Grad-Spanne je Achse (Rx, Ry, Rz) ueber den kompletten Pfad
#   - frac    : Verlauf 0..1 je Punkt (Default idx/(n-1); Punkt 5 von 10 -> 0.5)
#   Beim Aufruf sind 'idx' (0..n-1) und 'n' verfuegbar.
def parabola_waypoints(cfg):
    p = cfg.get("parabola", {})
    n = int(p.get("n", 20))
    y_max = float(p.get("y_max", 0.05))
    x0 = float(p.get("x0", 0.0))
    a = float(p.get("a", 1.0))
    z = float(p.get("z", 0.0))

    ori0 = cfg.get("ori0")
    if ori0 is None:
        ori0 = cfg["tcp_ori_deg"]
    scale = cfg.get("ori_scale", [0.0, 0.0, 0.0])

    wps = []
    for idx in range(n):
        value = -y_max + 2 * y_max * idx / (n - 1) if n > 1 else 0.0
        x = x0 + a * value * value
        frac = idx / (n - 1) if n > 1 else 0.0
        rot = [ori0[k] + scale[k] * frac for k in range(3)]
        wps.append({
            "name": f"W{idx}",
            "value": value,
            "tcp_pos": [x, value, z],
            "tcp_ori_deg": rot,
        })
    return wps


START_POSITIONS = {
    "aussen1": {
        "tcp_pos":  [0.615, 0, 0.295],
        "tcp_ori_deg": [180, 90, 0],
        "approach": [
            {"tcp_pos": [0.85, 0, 0.38], "tcp_ori_deg": [0, 0, 0], "label": "1", "use_current_seed": False},
            {"tcp_pos": [0.615, 0, 0.295], "tcp_ori_deg": [0, 90, 0], "label": "2", "use_current_seed": True},
        ],
        "jaw_pos":  [0.9, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "generator": parabola_waypoints,
        "parabola":  {"x0": 0.615, "a": 20, "z": 0.295, "n": 20, "y_max": 0.05},
        "ori_scale": [90, 30, 60],
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
        "parabola":  {"x0": 0.615, "a": 18.5, "z": 0.295, "n": 20, "y_max": 0.05},
        "ori_scale": [90, 30, 60],
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
