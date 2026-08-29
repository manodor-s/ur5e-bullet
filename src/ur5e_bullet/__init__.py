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


def _draw_waypoints(wps, active_idx=None):
    items = []
    color = [0.2, 0.8, 1.0]
    line_color = [0.2, 0.8, 1.0, 0.4]
    for i, wp in enumerate(wps):
        pos = wp["tcp_pos"]
        _draw_crosshair(pos, color, items)
        if i > 0:
            items.append(pybullet.addUserDebugLine(wps[i-1]["tcp_pos"], pos, line_color, 1))
    if len(wps) > 1:
        items.append(pybullet.addUserDebugLine(wps[-1]["tcp_pos"], wps[0]["tcp_pos"], line_color, 1))
    return items


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
    sim = UR5Sim()

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

    def draw_waypoints():
        for it in waypoint_items:
            try:
                pybullet.removeUserDebugItem(it)
            except Exception:
                pass
        waypoint_items.clear()
        if current_start is None:
            return
        wps = _resolve_waypoints(START_POSITIONS[current_start])
        if not wps:
            return
        waypoint_items.extend(_draw_waypoints(wps, waypoint_idx))

    def clear_temps():
        """Entfernt nur die temporaeren Vorschau- und TCP-Items.
        Das persistente Waypoint-Overlay (waypoint_items) bleibt davon unberuehrt,
        damit es nicht bei jeder Bewegung kurz verschwindet/neu gezeichnet wird."""
        for it in items:
            try:
                pybullet.removeUserDebugItem(it)
            except Exception:
                pass
        items.clear()
        for it in tcp_items:
            try:
                pybullet.removeUserDebugItem(it)
            except Exception:
                pass
        tcp_items.clear()

    def refresh_overlay():
        """Baut das TCP-Crosshair UND das Waypoint-Overlay nach einer Bewegung
        wieder auf. move_to schaltet das Rendering waehrend der Bewegung ab
        (configureDebugVisualizer COV_ENABLE_RENDERING) und wieder an; dabei
        verwirft pybullet die Debug-Items, daher muessen sie danach neu erzeugt
        werden. draw_waypoints() raeumt alte Items selbst auf (kein Doppel-Overlay)."""
        tcp_items.clear()
        tcp_items.extend(draw_tcp())
        draw_waypoints()

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

    def plan_and_show(pos, ori, seed=None):
        """Plant den RRT-Pfad (mit optionalem IK-Seed), zeichnet ihn kurz als
        Linie (ohne Rückfrage) und gibt den Joint-Pfad zum Fahren zurueck."""
        probe_result = sim._probe_path(pos, ori, seed=seed)
        start_tcp, _, _, _, rrt_waypoints, path = probe_result
        if rrt_waypoints:
            draw_probe_preview(rrt_waypoints, start_tcp, pos, ori)
            time.sleep(PREVIEW_PAUSE)
        else:
            print(f"  ⚠ Kein RRT-Pfad zu ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        clear_temps()
        return path

    items = []
    waypoint_items = []
    tcp_items = []

    if BOOT_START is not None:
        cfg = BOOT_START
        ori = [math.radians(v) for v in cfg["tcp_ori_deg"]]
        print("  Boot: fahre zur Startposition...")
        boot_path = plan_and_show(list(cfg["tcp_pos"]), ori, seed=sim._null_space[3])
        ok = sim.move_to(cfg["tcp_pos"], ori, speed=0.5, seed=sim._null_space[3], path=boot_path)
        if ok:
            print(f"  Boot: Startposition erreicht (Gebiss nicht geladen)")
        else:
            print("  Boot: Startposition nicht erreichbar – Arm bleibt an Neutralposition")
        tcp_items = draw_tcp()

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
    tcp_items = draw_tcp()

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
            clear_temps()
            continue
        if cmd.action == "reset" and sim._last_conf is not None:
            clear_temps()
            sim._execute([sim._last_conf])
            refresh_overlay()
            continue
        if cmd.action == "speed":
            current_speed = cmd.params["speed"]
            print(f"  Geschwindigkeit: {current_speed:.2f}")
            clear_temps()
            tcp_items = draw_tcp()
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
            tcp_items = draw_tcp()
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
                a_path = plan_and_show(a["tcp_pos"], a_ori, seed=a_seed)
                if a_path is None:
                    print(f"  ⛔ Approach {lbl} nicht erreichbar – start abgebrochen")
                    tcp_items = draw_tcp()
                    continue
                moved = sim.move_to(a["tcp_pos"], a_ori, speed=current_speed, seed=a_seed, path=a_path)
                if not moved:
                    print(f"  ⛔ Approach {lbl} nicht erreichbar – start abgebrochen")
                    tcp_items = draw_tcp()
                    continue
            print(f"  → fahre zu {name}-Start...")
            _dump_pose(f"final", cfg["tcp_pos"], tcp_ori)
            final_path = plan_and_show(cfg["tcp_pos"], tcp_ori, seed=start_seed)
            if final_path is None:
                print(f"  ⛔ {name}-Start nicht erreichbar – start abgebrochen")
                tcp_items = draw_tcp()
                continue
            moved = sim.move_to(cfg["tcp_pos"], tcp_ori, speed=current_speed, seed=start_seed, path=final_path)
            if not moved:
                print(f"  ⛔ {name}-Start nicht erreichbar – start abgebrochen")
                tcp_items = draw_tcp()
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
            waypoint_idx = 0
            wps = _resolve_waypoints(cfg)
            print(f"  Waypoints: {len(wps)}")
            for i, wp in enumerate(wps):
                print(f"    {i+1}: {wp.get('name') or wp.get('label', str(i+1))} ({wp['tcp_pos'][0]:.3f}, {wp['tcp_pos'][1]:.3f}, {wp['tcp_pos'][2]:.3f})")
            tcp_items = draw_tcp()
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
                if waypoint_idx >= len(wps):
                    print(f"  → Pfad-Ende ({len(wps)} Waypoints)")
                    break
                wp = wps[waypoint_idx]
                lbl = wp.get("name") or wp.get("label", str(waypoint_idx + 1))
                print(f"  → {current_start} {lbl} ({waypoint_idx+1}/{len(wps)})...")
                wp_ori = [math.radians(v) for v in wp["tcp_ori_deg"]]
                wp_path = plan_and_show(wp["tcp_pos"], wp_ori)
                if wp_path is None:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
                else:
                    moved = sim.move_to(wp["tcp_pos"], wp_ori, speed=current_speed, path=wp_path)
                    if not moved:
                        print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
                waypoint_idx += 1
            tcp_items = draw_tcp()
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
                wp_path = plan_and_show(wp["tcp_pos"], wp_ori)
                if wp_path is None:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
                else:
                    moved = sim.move_to(wp["tcp_pos"], wp_ori, speed=current_speed, path=wp_path)
                    if not moved:
                        print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
            tcp_items = draw_tcp()
            continue

        clear_temps()

        if cmd.action == "error":
            continue
        target_position = cmd.params["target_position"]
        target_orientation = cmd.params["target_orientation"]

        tcp_pos, _ = sim.get_tcp_pose()

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fds = (os.dup(1), os.dup(2))
        os.dup2(devnull_fd, 1), os.dup2(devnull_fd, 2)

        _draw_crosshair(target_position, [1, 1, 0], items,
                        f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

        probe_result = sim._probe_path(target_position, target_orientation)

        os.dup2(saved_fds[0], 1), os.dup2(saved_fds[1], 2)
        os.close(devnull_fd)

        start_tcp, _, _, _, rrt_waypoints, plan_path = probe_result

        target_pos, target_ori = draw_probe_preview(
            rrt_waypoints, start_tcp, target_position, target_orientation,
        )

        try:
            confirm = input("  Ausführen? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        ok = False
        if confirm in ("", "y", "yes"):
            ok = sim.move_to(target_pos, target_ori, speed=current_speed, path=plan_path)

        clear_temps()
        tcp_items = draw_tcp()

    if sim._mirror is not None:
        sim._mirror.close()

if __name__ == "__main__":
    demo_simulation()
