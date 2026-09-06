import os
import math
import time
import pybullet
from collections import namedtuple

from .sim import (
    UR5Sim,
    ROBOT_URDF_PATH,
    GEBISS_SCALE,
)

import importlib.util as _ilu
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.join(_pkg_dir, "..", "..")
_cfg_spec = _ilu.spec_from_file_location("config", os.path.join(_proj_root, "config.py"))
_cfg_mod = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
START_POSITIONS = _cfg_mod.START_POSITIONS
BOOT_START = _cfg_mod.BOOT_START
PREVIEW_PAUSE = _cfg_mod.PREVIEW_PAUSE
WAYPOINT_MARKER_RADIUS = _cfg_mod.WAYPOINT_MARKER_RADIUS
ENABLE_BLENDER_SYNC = _cfg_mod.ENABLE_BLENDER_SYNC
DRAW_VIEW_STICK = _cfg_mod.DRAW_VIEW_STICK
VIEW_STICK_LENGTH = _cfg_mod.VIEW_STICK_LENGTH
VIEW_STICK_RADIUS = _cfg_mod.VIEW_STICK_RADIUS
VIEW_STICK_COLOR = _cfg_mod.VIEW_STICK_COLOR
LOOK_TARGET_RADIUS = _cfg_mod.LOOK_TARGET_RADIUS
LOOK_TARGET_COLOR = _cfg_mod.LOOK_TARGET_COLOR
PB_CAMERA_DISTANCE = _cfg_mod.PB_CAMERA_DISTANCE
PB_CAMERA_YAW = _cfg_mod.PB_CAMERA_YAW
PB_CAMERA_PITCH = _cfg_mod.PB_CAMERA_PITCH
PB_CAMERA_TARGET_POS = _cfg_mod.PB_CAMERA_TARGET_POS


def _draw_crosshair(pos, color, items, label=None):
    h = 0.03
    items.append(pybullet.addUserDebugLine(
        [pos[0]-h, pos[1], pos[2]], [pos[0]+h, pos[1], pos[2]], color, 2,
    ))
    items.append(pybullet.addUserDebugLine(
        [pos[0], pos[1]-h, pos[2]], [pos[0], pos[1]+h, pos[2]], color, 2,
    ))
    items.append(pybullet.addUserDebugLine(
        [pos[0], pos[1], pos[2]-h], [pos[0], pos[1], pos[2]+h], color, 2,
    ))
    if label:
        items.append(pybullet.addUserDebugText(
            label, [pos[0], pos[1], pos[2]+0.04], color, 0.8,
        ))
    return items


Command = namedtuple("Command", [
    "action", "params",
])


def _parse_command(tokens):
    if tokens[0] == "q":
        return Command("quit", {})
    if tokens[0] == "o" and len(tokens) == 4:
        try:
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        except ValueError:
            print(f"  ? '{' '.join(tokens)}' verstanden?")
            return Command("error", {})
        return Command("offset", {"pos": pos})
    if tokens[0] == "s" and len(tokens) == 2:
        try:
            speed = float(tokens[1])
        except ValueError:
            print(f"  ? '{' '.join(tokens)}' verstanden?")
            return Command("error", {})
        return Command("speed", {"speed": max(speed, 0.01)})
    if tokens[0] == "@":
        return Command("reset", {})
    if tokens[0] == "render":
        return Command("render", {})
    if tokens[0] == "jaw":
        if len(tokens) < 2:
            print("  ? 'jaw <nr> [upper|lower]' erwartet")
            return Command("error", {})
        try:
            folder = int(tokens[1])
        except ValueError:
            print(f"  ? '{tokens[1]}' ist keine gueltige Nr.")
            return Command("error", {})
        jaw_type = tokens[2] if len(tokens) >= 3 else "lower"
        if jaw_type not in ("upper", "lower"):
            print(f"  ? '{jaw_type}' – nur 'upper' oder 'lower'")
            return Command("error", {})
        return Command("jaw", {"folder": folder, "type": jaw_type})
    if tokens[0] == "start":
        if len(tokens) < 2:
            names = ", ".join(START_POSITIONS.keys())
            print(f"  ? Verwende: start <{names}>")
            return Command("error", {})
        name = tokens[1]
        if name not in START_POSITIONS:
            print(f"  ? '{name}' – verfuegbar: {', '.join(START_POSITIONS.keys())}")
            return Command("error", {})
        return Command("start_pos", {"name": name})
    if tokens[0] == "+":
        n = 1
        if len(tokens) >= 2:
            try:
                n = int(tokens[1])
            except ValueError:
                print(f"  ? '{tokens[1]}' ist keine Zahl")
                return Command("error", {})
        return Command("waypoint_next", {"steps": n})
    if tokens[0] == "-":
        n = 1
        if len(tokens) >= 2:
            try:
                n = int(tokens[1])
            except ValueError:
                print(f"  ? '{tokens[1]}' ist keine Zahl")
                return Command("error", {})
        return Command("waypoint_prev", {"steps": n})

    if tokens[0] != "m":
        print(f"  ? '{' '.join(tokens)}' verstanden? (Move: 'm x y z [rx ry rz]')")
        return Command("error", {})
    try:
        parsed_values = [float(v) for v in tokens[1:]]
    except ValueError:
        print(f"  ? '{' '.join(tokens)}' verstanden? (Move: 'm x y z [rx ry rz]')")
        return Command("error", {})
    if len(parsed_values) < 3:
        print(f"  ? '{' '.join(tokens)}' – Position 'x y z' fehlt")
        return Command("error", {})
    target_position = parsed_values[:3]
    target_orientation = [math.radians(v) for v in parsed_values[3:6]] if len(parsed_values) >= 6 else [0, 0, 0]
    return Command("move", {
        "target_position": target_position,
        "target_orientation": target_orientation,
    })


def _draw_waypoint_bodies(wps):
    """Erzeugt fuer jeden Waypoint einen kleinen, persistenten Kugel-Body.

    Waypoints werden als echte pybullet-Bodies statt als Debug-Items dargestellt:
    Damit sind sie von removeAllUserDebugItems() und der Debug-Handle-Invalidierung
    (die zu Geister-Resten fuehrt) vollstaendig unberuehrt. Sie bleiben stehen, bis
    die Bodies explizit per removeBody() entfernt werden -> kein staendiges
    Neuzeichnen. Kugeln erhalten KEINE Kollisionsgeometrie, daher kollidieren sie
    nie mit dem Roboter und zaehlen nicht als Hindernisse."""
    bodies = []
    for wp in wps:
        pos = wp["tcp_pos"]
        vis = pybullet.createVisualShape(
            pybullet.GEOM_SPHERE, radius=WAYPOINT_MARKER_RADIUS, rgbaColor=[0.2, 0.8, 1.0, 0.9],
        )
        body = pybullet.createMultiBody(
            baseVisualShapeIndex=vis, basePosition=pos,
        )
        bodies.append(body)
    return bodies


def _draw_look_target_body(pos):
    """Persistenter, kollisionsfreier Marker fuer das Blickachse-Ziel (Look-Target).
    Wie die Waypoints: nur visuelle Geometrie, kollidiert nie mit dem Roboter."""
    vis = pybullet.createVisualShape(
        pybullet.GEOM_SPHERE, radius=LOOK_TARGET_RADIUS, rgbaColor=LOOK_TARGET_COLOR,
    )
    return pybullet.createMultiBody(baseVisualShapeIndex=vis, basePosition=pos)


def _resolve_waypoints(cfg):
    """Erzeugt die Liste der Waypoint-Dicts einer Startposition.

    - 'generator' : echte Python-Funktion (in config.py definiert), die aus
                    der Startposition-CFG Position UND Orientierung berechnet.
    - 'waypoints' : fallback auf eine feste Liste (dicts mit tcp_pos/
                    tcp_ori_deg), wie bisher.
    """
    gen = cfg.get("generator")
    if callable(gen):
        return gen(cfg)
    return cfg.get("waypoints", [])


def demo_simulation():
    sim = UR5Sim(mirror=ENABLE_BLENDER_SYNC)

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

    def _remove_view_stick():
        nonlocal view_stick_id
        if view_stick_id is not None:
            try:
                pybullet.removeBody(view_stick_id)
            except Exception:
                pass
            view_stick_id = None

    def draw_view_stick():
        """Zeichnet einen kollisionsfreien Stab vom TCP aus in Richtung der
        Kamera-Blickachse. Stab ist ein persistenter Multibody (ueberlebt
        removeAllUserDebugItems) und wird bei jeder Bewegung repositioniert."""
        nonlocal view_stick_id
        if not DRAW_VIEW_STICK:
            _remove_view_stick()
            return
        pos, quat = sim.get_tcp_pose()
        R = pybullet.getMatrixFromQuaternion(quat)
        # Kamera-Blickachse (Kandidat: TCP-lokales -Z) -> Weltrichtung
        view = [-(R[2]), -(R[5]), -(R[8])]
        length = VIEW_STICK_LENGTH
        center = [pos[i] + view[i] * length / 2 for i in range(3)]
        z = (0.0, 0.0, 1.0)
        dot = z[0]*view[0] + z[1]*view[1] + z[2]*view[2]
        ang = math.acos(max(-1.0, min(1.0, dot)))
        axis = [
            z[1]*view[2] - z[2]*view[1],
            z[2]*view[0] - z[0]*view[2],
            z[0]*view[1] - z[1]*view[0],
        ]
        n = math.sqrt(sum(c*c for c in axis))
        oq = (0.0, 0.0, 0.0, 1.0)
        if n > 1e-6:
            oq = pybullet.getQuaternionFromAxisAngle([c / n for c in axis], ang)
        if view_stick_id is None:
            vis = pybullet.createVisualShape(
                pybullet.GEOM_CYLINDER,
                radius=VIEW_STICK_RADIUS,
                length=length,
                rgbaColor=VIEW_STICK_COLOR,
            )
            view_stick_id = pybullet.createMultiBody(
                baseVisualShapeIndex=vis,
                basePosition=center,
                baseOrientation=oq,
            )
        else:
            pybullet.resetBasePositionAndOrientation(view_stick_id, center, oq)

    def draw_waypoints():
        nonlocal look_target_body_id
        for b in waypoint_bodies:
            try:
                pybullet.removeBody(b)
            except Exception:
                pass
        waypoint_bodies.clear()
        if look_target_body_id is not None:
            try:
                pybullet.removeBody(look_target_body_id)
            except Exception:
                pass
            look_target_body_id = None
        if current_start is None:
            return
        cfg = START_POSITIONS[current_start]
        wps = _resolve_waypoints(cfg)
        if not wps:
            return
        waypoint_bodies.extend(_draw_waypoint_bodies(wps))
        plane_z = wps[0]["tcp_pos"][2]
        lt = cfg.get("look_target")
        jaw = cfg.get("jaw_pos")
        if lt is not None:
            target = [lt[0], lt[1], plane_z]
        elif jaw is not None:
            target = [jaw[0], jaw[1], plane_z]
        else:
            target = None
        if target is not None:
            look_target_body_id = _draw_look_target_body(target)

    def _set_view(cfg):
        """Setzt die pybullet-Orbit-Kamera auf die Startposition/den Startwinkel
        der Startposition (cfg['view']). Fehlt 'view' oder 'apply' ist False,
        wird die Kamera nicht beruehrt (Global aus sim.py gilt dann)."""
        v = cfg.get("view")
        if not v or not v.get("apply", True):
            return
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        pybullet.resetDebugVisualizerCamera(
            v.get("distance", PB_CAMERA_DISTANCE),
            v.get("yaw", PB_CAMERA_YAW),
            v.get("pitch", PB_CAMERA_PITCH),
            list(v.get("target", PB_CAMERA_TARGET_POS)),
        )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def reset_overlay():
        """Entfernt ALLE Debug-Items (auch verwaiste "Geister") via
        removeAllUserDebugItems und zeichnet das TCP-Crosshair neu.
        Die persistenten Waypoint-Bodies sind davon unberuehrt und bleiben stehen."""
        nonlocal tcp_items
        pybullet.removeAllUserDebugItems()
        items.clear()
        tcp_items.clear()
        tcp_items = draw_tcp()
        draw_view_stick()

    def draw_probe_preview(rrt_waypoints, start_tcp, target_position, target_orientation):
        if rrt_waypoints:
            step = max(1, len(rrt_waypoints) // 20)
            for i in range(0, len(rrt_waypoints)-step, step):
                items.append(pybullet.addUserDebugLine(
                    rrt_waypoints[i], rrt_waypoints[i+step], [0, 1, 0], 1,
                ))
        else:
            print(f"  ⚠ Kein RRT-Pfad zu ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
        return target_position, target_orientation

    def preview_and_move(pos, ori, speed, seed=None, confirm=False):
        """Gemeinsamer Ablauf fuer den m-Befehl (confirm=True, mit Rueckfrage)
        und alle automatischen Bewegungen (confirm=False, kurze Pause):
        Zielmarker + RRT-Pfad zeichnen, warten, den Pfad (1x) fahren, aufraeumen.
        Gibt True bei Erfolg, False wenn kein Pfad/unerreichbar, None bei Abbruch."""
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fds = (os.dup(1), os.dup(2))
        os.dup2(devnull_fd, 1), os.dup2(devnull_fd, 2)

        probe_result = sim._probe_path(pos, ori, seed=seed)

        os.dup2(saved_fds[0], 1), os.dup2(saved_fds[1], 2)
        os.close(devnull_fd)

        start_tcp, _, _, _, rrt_waypoints, plan_path = probe_result
        _draw_crosshair(pos, [1, 1, 0], items,
                        f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        target_pos, target_ori = draw_probe_preview(
            rrt_waypoints, start_tcp, pos, ori,
        )
        ok = False
        if plan_path is not None:
            if confirm:
                try:
                    c = input("  Ausführen? [Y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    reset_overlay()
                    return None
                if c in ("", "y", "yes"):
                    ok = sim.move_to(target_pos, target_ori, speed=speed, path=plan_path)
            else:
                time.sleep(PREVIEW_PAUSE)
                ok = sim.move_to(target_pos, target_ori, speed=speed, path=plan_path)
        reset_overlay()
        return ok

    items = []
    waypoint_bodies = []
    look_target_body_id = None
    tcp_items = []
    view_stick_id = None
    current_start = None
    waypoint_idx = 0

    if BOOT_START is not None:
        cfg = BOOT_START
        ori = [math.radians(v) for v in cfg["tcp_ori_deg"]]
        print("  Boot: fahre zur Startposition...")
        ok = preview_and_move(list(cfg["tcp_pos"]), ori, 0.5, seed=sim._null_space[3])
        if ok:
            print(f"  Boot: Startposition erreicht (Gebiss nicht geladen)")
        else:
            print("  Boot: Startposition nicht erreichbar – Arm bleibt an Neutralposition")
        _set_view(cfg)

    print("── UR5e Demo ──────────────────────────────")
    print("Move:        'm x y z [rx ry rz]' (RRT, Default-Orientierung 0 0 0)")
    print("  Tool-Offset: 'o x y z' (z. B. o 0 0 0.15)")
    print("  Geschw.:     's 0.5' (global, Default 0.5)")
    print("  Reset:       '@'  (nach manuellem Ziehen)")
    print("  Render:      'render' (Cycles-Render in Blender)")
    print("  Gebiss:      'jaw <nr> [upper|lower]' (z. B. jaw 3 upper)")
    print("  Start:       'start <Aussen|Oben|Innen>' (Startposition anfahren)")
    print("  Waypoints:   '+'/'-' naechster/vorheriger Waypoint")
    print("────────────────────────────────────────────")
    current_start = None
    waypoint_idx = 0
    current_speed = 0.5
    reset_overlay()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        tokens = line.split()
        cmd = _parse_command(tokens)
        if cmd.action == "quit":
            break
        if cmd.action == "offset":
            sim.set_tool_offset(cmd.params["pos"])
            reset_overlay()
            continue
        if cmd.action == "reset" and sim._last_conf is not None:
            sim._execute([sim._last_conf])
            reset_overlay()
            continue
        if cmd.action == "speed":
            current_speed = cmd.params["speed"]
            print(f"  Geschwindigkeit: {current_speed:.2f}")
            reset_overlay()
            continue
        if cmd.action == "render":
            sim._mirror._render_done.clear()
            sim._mirror.send_message({"render": True})
            print("  Render gestartet...")
            sim._mirror._render_done.wait(timeout=300)
            continue
        if cmd.action == "jaw":
            if current_start is None:
                print(f"  ? Keine Startposition aktiv – zuerst 'start <name>'")
                continue
            folder = cmd.params["folder"]
            jaw_type = cmd.params["type"]
            jcfg = START_POSITIONS[current_start]
            jpos = jcfg["jaw_pos"]
            jeuler = [math.radians(v) for v in jcfg["jaw_euler_deg"]]
            try:
                ok = sim.load_jaw(folder, jaw_type, jpos, jeuler)
            except FileNotFoundError as e:
                print(f"  ! {e}")
                continue
            if not ok:
                continue
            print(f"  PyBullet: gebiss_{jaw_type} aus Ordner {folder}")
            if sim._mirror is not None:
                sim._mirror._jaw_done.clear()
                sim._mirror.send_message({"replace_jaw": {"folder": folder, "type": jaw_type, "pos": jpos, "euler": jcfg["jaw_euler_deg"]}})
                sim._mirror._jaw_done.wait(timeout=10)
            reset_overlay()
            continue
        if cmd.action == "start_pos":
            name = cmd.params["name"]
            cfg = START_POSITIONS[name]
            tcp_ori = [math.radians(v) for v in cfg["tcp_ori_deg"]]
            print(f"  → jaw entfernt...")
            sim.unload_jaw()
            if sim._mirror is not None:
                sim._mirror.send_message({"jaw_unload": True})

            def _dump_pose(tag, pos, ori_deg):
                q = pybullet.getQuaternionFromEuler(ori_deg)
                ee = sim._tcp_to_ee(list(pos), list(ori_deg))
                joints = sim.get_joint_angles()
                jstr = ", ".join(f"{math.degrees(j):.0f}" for j in joints)
                print(f"    [dbg {tag}] tcp=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) euler_deg=({math.degrees(ori_deg[0]):.0f},{math.degrees(ori_deg[1]):.0f},{math.degrees(ori_deg[2]):.0f})")
                print(f"    [dbg {tag}] quat=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})")
                print(f"    [dbg {tag}] ee_pos=({ee[0][0]:.3f},{ee[0][1]:.3f},{ee[0][2]:.3f})")
                print(f"    [dbg {tag}] joints_deg={jstr}")

            approach = cfg.get("approach", [])
            start_seed = sim._null_space[3]
            for i, a in enumerate(approach):
                a_ori = [math.radians(v) for v in a["tcp_ori_deg"]]
                lbl = a.get("label", str(i + 1))
                print(f"  → approach {lbl}...")
                _dump_pose(f"approach{lbl}", a["tcp_pos"], a_ori)
                a_seed = None if a.get("use_current_seed") else start_seed
                ok = preview_and_move(a["tcp_pos"], a_ori, current_speed, seed=a_seed)
                if not ok:
                    print(f"  ⛔ Approach {lbl} nicht erreichbar – start abgebrochen")
                    continue
            print(f"  → fahre zu {name}-Start...")
            _dump_pose(f"final", cfg["tcp_pos"], tcp_ori)
            ok = preview_and_move(cfg["tcp_pos"], tcp_ori, current_speed, seed=start_seed)
            if not ok:
                print(f"  ⛔ {name}-Start nicht erreichbar – start abgebrochen")
                continue
            jpos = cfg["jaw_pos"]
            jeuler = [math.radians(v) for v in cfg["jaw_euler_deg"]]
            sim.load_jaw_at(cfg["jaw_folder"], cfg["jaw_type"], jpos, jeuler)
            if sim._mirror is not None:
                sim._mirror._jaw_done.clear()
                sim._mirror.send_message({"replace_jaw": {"folder": cfg["jaw_folder"], "type": cfg["jaw_type"], "pos": jpos, "euler": cfg["jaw_euler_deg"]}})
                sim._mirror._jaw_done.wait(timeout=10)
                sim._mirror.send_current()
            print(f"  → jaw eingefuegt ({cfg['jaw_type']}, pos=({jpos[0]:.3f}, {jpos[1]:.3f}, {jpos[2]:.3f}))")
            current_start = name
            waypoint_idx = 10
            wps = _resolve_waypoints(cfg)
            print(f"  Waypoints: {len(wps)}")
            for i, wp in enumerate(wps):
                print(f"    {i+1}: {wp.get('name') or wp.get('label', str(i+1))} ({wp['tcp_pos'][0]:.3f}, {wp['tcp_pos'][1]:.3f}, {wp['tcp_pos'][2]:.3f})")
            _set_view(cfg)
            reset_overlay()
            draw_waypoints()
            continue
        if cmd.action == "waypoint_next":
            if current_start is None:
                print(f"  ? Keine Startposition aktiv (zuerst 'start <name>')")
                continue
            cfg = START_POSITIONS[current_start]
            wps = _resolve_waypoints(cfg)
            if not wps:
                print(f"  ? Keine Waypoints definiert fuer {current_start}")
                continue
            steps = cmd.params["steps"]
            for s in range(steps):
                if waypoint_idx + 1 >= len(wps):
                    print(f"  → Pfad-Ende ({len(wps)} Waypoints)")
                    break
                waypoint_idx += 1
                wp = wps[waypoint_idx]
                lbl = wp.get("name") or wp.get("label", str(waypoint_idx + 1))
                print(f"  → {current_start} {lbl} ({waypoint_idx+1}/{len(wps)})...")
                wp_ori = [math.radians(v) for v in wp["tcp_ori_deg"]]
                ok = preview_and_move(wp["tcp_pos"], wp_ori, current_speed)
                if not ok:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
            reset_overlay()
            continue
        if cmd.action == "waypoint_prev":
            if current_start is None:
                print(f"  ? Keine Startposition aktiv (zuerst 'start <name>')")
                continue
            cfg = START_POSITIONS[current_start]
            wps = _resolve_waypoints(cfg)
            if not wps:
                print(f"  ? Keine Waypoints definiert fuer {current_start}")
                continue
            steps = cmd.params["steps"]
            for s in range(steps):
                if waypoint_idx <= 0:
                    print(f"  → Startposition erreicht")
                    break
                waypoint_idx -= 1
                wp = wps[waypoint_idx]
                lbl = wp.get("name") or wp.get("label", str(waypoint_idx + 1))
                print(f"  → {current_start} {lbl} ({waypoint_idx+1}/{len(wps)}) zurueck...")
                wp_ori = [math.radians(v) for v in wp["tcp_ori_deg"]]
                ok = preview_and_move(wp["tcp_pos"], wp_ori, current_speed)
                if not ok:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
            reset_overlay()
            continue

        if cmd.action == "error":
            continue
        target_position = cmd.params["target_position"]
        target_orientation = cmd.params["target_orientation"]

        result = preview_and_move(target_position, target_orientation, current_speed, confirm=True)
        if result is None:
            break

    if sim._mirror is not None:
        sim._mirror.close()

if __name__ == "__main__":
    demo_simulation()
