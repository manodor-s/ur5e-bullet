import math
import os

import pybullet as pb

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
CAMERA_FAR_M = 0.03
CAMERA_DISPLAY_M = 0.2

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
# Position (power-Parabel): x = x0 + a*|value/y_max|^power,  y = value,  z = z
#   - power : Kurvenform; 2 = flache x^2-Parabel, 4 = x^4 (flache Mitte,
#             steile Enden, Gebissform), auch 3 oder andere Werte moeglich.
#   - Die Renormierung (value/y_max) macht 'a' direkt zum x-Ausschlag an den
#     aeussersten Wegpunkten (W0/W{n-1}): dort ist |value/y_max|^power = 1
#     -> x = x0 + a, unabhaengig von power und y_max.
#   - Gleiche Verteilung: Die n Wegpunkte werden NICHT gleichmaessig in 'value',
#     sondern gleichmaessig nach Bogenlaenge entlang der Kurve verteilt
#     (konstanter Abstand auf der Linie). Dadurch sammeln sich die Punkte nicht
#     in der Mitte, sondern liegen gleichmaessig auf der Kurve.
#
# Orientierung: smoother Uebergang (Quaternion-Slerp) zwischen drei Ankern.
#   ori_anchors = {"start": <euler>, "mid": <euler>, "end": <euler>}
#   - start : Orientierung am ersten Waypoint (W0)
#   - mid   : Orientierung am mittigen Waypoint (n muss UNGERADE sein, damit
#             der mittige Exakt mittig liegt); mittiger Index = n//2
#   - end   : Orientierung am letzten Waypoint (W{n-1})
#   Alle Winkel in Grad. Zwischenpunkte werden stueckweise sph aerisch
#   linear interpoliert (slerp), dadurch werden Achsenspruenge/Mehrdeutigkeiten
#   der Euler-Darstellung vermieden.
def _quat_normalize(q):
    import math as _m
    n = _m.sqrt(sum(c * c for c in q))
    return [c / n for c in q] if n > 0 else [1.0, 0.0, 0.0, 0.0]


def _quat_slerp(a, b, t):
    import math as _m
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = [-c for c in b]
        dot = -dot
    if dot > 0.9995:
        r = [a[i] + t * (b[i] - a[i]) for i in range(4)]
        return _quat_normalize(r)
    theta = _m.acos(min(1.0, dot))
    so = _m.sin(theta)
    wa = _m.sin((1.0 - t) * theta) / so
    wb = _m.sin(t * theta) / so
    return [wa * a[i] + wb * b[i] for i in range(4)]


def parabola_waypoints(cfg):
    p = cfg.get("parabola", {})
    n = int(p.get("n", 20))
    y_max = float(p.get("y_max", 0.05))
    x0 = float(p.get("x0", 0.0))
    a = float(p.get("a", 1.0))
    z = float(p.get("z", 0.0))
    power = float(p.get("power", 4.0))

    anchors = cfg.get("ori_anchors", {})
    a_start = anchors.get("start", [90, 0, 0])
    a_mid = anchors.get("mid", [180, 90, 0])
    a_end = anchors.get("end", [-90, 0, 0])
    q_start = pb.getQuaternionFromEuler([math.radians(v) for v in a_start])
    q_mid = pb.getQuaternionFromEuler([math.radians(v) for v in a_mid])
    q_end = pb.getQuaternionFromEuler([math.radians(v) for v in a_end])

    mid = n // 2
    last = n - 1

    # Bogenlaenge entlang der Kurve numerisch berechnen: v -> kumulierte Laenge.
    # x(v) = x0 + a*|v/y_max|^power, y(v) = v  -> ds/dv = sqrt((dx/dv)^2 + 1)
    def dx_dv(v):
        return a * power * abs(v / y_max) ** (power - 1) / y_max if y_max > 0 else 0.0

    n_int = 4000
    v_grid = []
    L_grid = []
    acc = 0.0
    prev = -y_max
    prev_x = dx_dv(prev)
    v_grid.append(prev)
    L_grid.append(0.0)
    for k in range(1, n_int + 1):
        v = -y_max + 2 * y_max * k / n_int
        d = dx_dv(v)
        acc += math.sqrt(((prev_x + d) / 2.0) ** 2 + 1.0) * (v - prev)
        prev = v
        prev_x = d
        v_grid.append(v)
        L_grid.append(acc)
    L_total = acc

    if n > 1 and L_total > 0:
        values = []
        for i in range(n):
            target = L_total * i / (n - 1)
            # invertiere L(v)=target per linearer Interpolation (L monoton)
            for k in range(1, len(L_grid)):
                if L_grid[k] >= target:
                    v0, v1 = v_grid[k - 1], v_grid[k]
                    l0, l1 = L_grid[k - 1], L_grid[k]
                    if l1 > l0:
                        f = (target - l0) / (l1 - l0)
                    else:
                        f = 0.0
                    values.append(v0 + f * (v1 - v0))
                    break
            else:
                values.append(v_grid[-1])
    else:
        values = [0.0] * n

    wps = []
    for idx in range(n):
        value = values[idx]
        x = x0 + a * abs(value / y_max) ** power if y_max > 0 else x0
        if idx <= mid:
            t = idx / mid if mid > 0 else 0.0
            q = _quat_slerp(q_start, q_mid, t)
        else:
            t = (idx - mid) / (last - mid) if last > mid else 1.0
            q = _quat_slerp(q_mid, q_end, t)
        rot = [math.degrees(v) for v in pb.getEulerFromQuaternion(q)]
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
        "jaw_pos":  [0.65, 0, 0.3],
        "jaw_euler_deg": [0, 0, 90],
        "jaw_folder": 1,
        "jaw_type":  "lower",
        "generator": parabola_waypoints,
        "parabola":  {"x0": 0.615, "a": 0.046, "z": 0.295, "n": 21, "y_max": 0.037, "power": 4},
        "ori_anchors": {"start": [90, 0, 0], "mid": [180, 90, 0], "end": [-90, 0, 0]},
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
