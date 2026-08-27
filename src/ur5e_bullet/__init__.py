import os
import math
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

    linear = True
    idx = 0
    if tokens[0] == "r":
        linear = False
        idx = 1
    try:
        parsed_values = [float(v) for v in tokens[idx:]]
    except ValueError:
        print(f"  ? '{' '.join(tokens)}' verstanden?")
        return Command("error", {})
    if len(parsed_values) < 3:
        return Command("incomplete", {})
    target_position = parsed_values[:3]
    target_orientation = [math.radians(v) for v in parsed_values[3:6]] if len(parsed_values) >= 6 else [0, 0, 0]
    return Command("move", {
        "target_position": target_position,
        "target_orientation": target_orientation,
        "linear": linear,
    })


def _draw_waypoints(wps, active_idx=None):
    items = []
    color = [0.2, 0.8, 1.0]
    line_color = [0.2, 0.8, 1.0, 0.4]
    for i, wp in enumerate(wps):
        pos = wp["tcp_pos"]
        _draw_crosshair(pos, color, items, wp.get("label", str(i)))
        if i > 0:
            items.append(pybullet.addUserDebugLine(wps[i-1]["tcp_pos"], pos, line_color, 1))
    if len(wps) > 1:
        items.append(pybullet.addUserDebugLine(wps[-1]["tcp_pos"], wps[0]["tcp_pos"], line_color, 1))
    return items


def _resolve_waypoints(cfg):
    """Erzeugt die Liste der Waypoint-Dicts aus einer parametrischen Spezifikation.

    Zwei Varianten, erkennbar am Key der Startposition:
      - 'waypoints': feste Liste (dicts mit tcp_pos/tcp_ori_deg), wie bisher.
      - 'parabola' : Parabel  x = x0 + a * value^2, z = z. Die 'values'-Liste
                     gibt je Punkt den Verlaufsparameter (hier y) vor; Position
                     und Orientierung werden daraus berechnet.
    Optionale 'ori_func' je Startposition: Python-Ausdruck (String) oder
    Callable, das aus dem Wert (Variable 'value') die TCP-Orientierung
    [rx, ry, rz] in Grad liefert. Ohne ori_func gilt Default [0, 0, 90].
    """
    if "parabola" in cfg:
        par = cfg["parabola"]
        x0 = par.get("x0", 0.0)
        a = par.get("a", 1.0)
        z = par.get("z", 0.0)
        y_max = par.get("y_max")
        n = par.get("n", 20)
        values = cfg.get("values")
        if not values:
            if y_max is None:
                values = [float(i) for i in range(n)]
            else:
                values = [-y_max + 2 * y_max * i / (n - 1) for i in range(n)]
        ori_func = cfg.get("ori_func")
        wps = []
        for j, value in enumerate(values):
            v = float(value)
            pos = [x0 + a * v * v, v, z]
            o = _resolve_ori(ori_func, value)
            wps.append({"tcp_pos": pos, "tcp_ori_deg": o, "label": str(j), "value": value})
        return wps
    return cfg.get("waypoints", [])


def _resolve_ori(ori_func, value):
    if ori_func is None:
        return [0.0, 0.0, 90.0]
    if callable(ori_func):
        return list(ori_func(value))
    g = {"value": value}
    result = eval(ori_func, {"__builtins__": {}}, g)
    return [float(x) for x in result]


def demo_simulation():
    sim = UR5Sim()

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

    def draw_waypoints():
        if current_start is None:
            return []
        wps = _resolve_waypoints(START_POSITIONS[current_start])
        if not wps:
            return []
        return _draw_waypoints(wps, waypoint_idx)

    def draw_probe_preview(last_valid_position, collision_position, deviation_mm,
                           rrt_waypoints, start_tcp, target_position, target_orientation,
                           linear):
        if collision_position is None and deviation_mm == 0.0:
            if rrt_waypoints:
                step = max(1, len(rrt_waypoints) // 20)
                for i in range(0, len(rrt_waypoints)-step, step):
                    items.append(pybullet.addUserDebugLine(
                        rrt_waypoints[i], rrt_waypoints[i+step], [0, 1, 0], 1,
                    ))
            else:
                items.append(pybullet.addUserDebugLine(
                    start_tcp, target_position, [0, 1, 0], 2,
                ))
            return target_position, target_orientation

        if collision_position is None and deviation_mm > 0:
            items.append(pybullet.addUserDebugLine(
                start_tcp, last_valid_position, [0, 1, 0], 2,
            ))
            items.append(pybullet.addUserDebugLine(
                last_valid_position, target_position, [1, 0, 0], 2,
            ))
            reachable_distance = math.sqrt(sum((last_valid_position[i]-start_tcp[i])**2 for i in range(3)))
            if reachable_distance > 0.005:
                target_distance = math.sqrt(sum((target_position[i]-start_tcp[i])**2 for i in range(3)))
                slerp_fraction = reachable_distance / target_distance if target_distance > 0 else 0
                _, current_orientation = sim.get_tcp_pose()
                mid_orientation = pybullet.getQuaternionSlerp(
                    current_orientation, pybullet.getQuaternionFromEuler(target_orientation), slerp_fraction,
                )
                print(f"  ⚠ {deviation_mm:.0f}mm Abweichung – fahre zu ({last_valid_position[0]:.3f}, {last_valid_position[1]:.3f}, {last_valid_position[2]:.3f})")
                return last_valid_position, list(pybullet.getEulerFromQuaternion(mid_orientation))
            else:
                _draw_crosshair(target_position, [1, 0, 0], items)
                print(f"  ⛔ Kein linearer Pfad zu ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
                return target_position, target_orientation

        items.append(pybullet.addUserDebugLine(
            start_tcp, last_valid_position, [0, 1, 0], 2,
        ))
        items.append(pybullet.addUserDebugLine(
            last_valid_position, target_position, [1, 0, 0], 2,
        ))
        _draw_crosshair(last_valid_position, [1, 0, 0], items)
        if not linear:
            print(f"  ⛔ Kein RRT-Pfad zu ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
        else:
            err = sim._last_linear_error
            if err and err[0] == "limit":
                names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
                print(f"  ⛔ {names[err[2]]} am Limit")
            else:
                print(f"  ⛔ Kollision bei ({collision_position[0]:.3f}, {collision_position[1]:.3f}, {collision_position[2]:.3f})")
        return target_position, target_orientation

    if BOOT_START is not None:
        cfg = BOOT_START
        ori = [math.radians(v) for v in cfg["tcp_ori_deg"]]
        print("  Boot: fahre zur Startposition...")
        ok = sim.move_to(cfg["tcp_pos"], ori, linear=True, speed=0.5)
        if not ok:
            ok = sim.move_to(cfg["tcp_pos"], ori, linear=False, speed=0.5)
        if ok:
            print(f"  Boot: Startposition erreicht (Gebiss nicht geladen)")
        else:
            print("  Boot: Startposition nicht erreichbar – Arm bleibt an Neutralposition")
        tcp_items = draw_tcp()

    print("── UR5e Demo ──────────────────────────────")
    print("Format: x y z rx ry rz oder 'q' zum Beenden")
    print("  Tool-Offset: 'o x y z' (z. B. o 0 0 0.15)")
    print("  RRT-Modus:   'r x y z rx ry rz'")
    print("  Geschw.:     's 0.5' (global, Default 0.5)")
    print("  Reset:       '@'  (nach manuellem Ziehen)")
    print("  Render:      'render' (Cycles-Render in Blender)")
    print("  Gebiss:      'jaw <nr> [upper|lower]' (z. B. jaw 3 upper)")
    print("  Start:       'start <Aussen|Oben|Innen>' (Startposition anfahren)")
    print("  Waypoints:   '+'/'-' naechster/vorheriger Waypoint")
    print("────────────────────────────────────────────")
    items = []
    waypoint_items = []
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
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            waypoint_items = draw_waypoints()
            continue
        if cmd.action == "reset" and sim._last_conf is not None:
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            sim._execute([sim._last_conf])
            tcp_items = draw_tcp()
            waypoint_items = draw_waypoints()
            continue
        if cmd.action == "speed":
            current_speed = cmd.params["speed"]
            print(f"  Geschwindigkeit: {current_speed:.2f}")
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            tcp_items = draw_tcp()
            waypoint_items = draw_waypoints()
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
            for i, a in enumerate(approach):
                a_ori = [math.radians(v) for v in a["tcp_ori_deg"]]
                lbl = a.get("label", str(i + 1))
                print(f"  → approach {lbl}...")
                _dump_pose(f"approach{lbl}", a["tcp_pos"], a_ori)
                moved = sim.move_to(a["tcp_pos"], a_ori, linear=True, speed=current_speed)
                if not moved:
                    moved = sim.move_to(a["tcp_pos"], a_ori, linear=False, speed=current_speed)
                if not moved:
                    print(f"  ⛔ Approach {lbl} nicht erreichbar – start abgebrochen")
                    tcp_items = draw_tcp()
                    continue
            print(f"  → fahre zu {name}-Start...")
            _dump_pose(f"final", cfg["tcp_pos"], tcp_ori)
            moved = sim.move_to(cfg["tcp_pos"], tcp_ori, linear=True, speed=current_speed)
            if not moved:
                moved = sim.move_to(cfg["tcp_pos"], tcp_ori, linear=False, speed=current_speed)
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
                print(f"    {i+1}: {wp.get('label', str(i+1))} ({wp['tcp_pos'][0]:.3f}, {wp['tcp_pos'][1]:.3f}, {wp['tcp_pos'][2]:.3f})")
            tcp_items = draw_tcp()
            waypoint_items = draw_waypoints()
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
                lbl = wp.get("label", str(waypoint_idx + 1))
                print(f"  → {current_start} {lbl} ({waypoint_idx+1}/{len(wps)})...")
                pybullet.removeAllUserDebugItems()
                items.clear()
                tcp_items = []
                wp_ori = [math.radians(v) for v in wp["tcp_ori_deg"]]
                moved = sim.move_to(wp["tcp_pos"], wp_ori, linear=True, speed=current_speed)
                if not moved:
                    moved = sim.move_to(wp["tcp_pos"], wp_ori, linear=False, speed=current_speed)
                if not moved:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
                waypoint_idx += 1
            tcp_items = draw_tcp()
            waypoint_items = draw_waypoints()
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
                lbl = wp.get("label", str(waypoint_idx + 1))
                print(f"  → {current_start} {lbl} ({waypoint_idx+1}/{len(wps)}) zurueck...")
                pybullet.removeAllUserDebugItems()
                items.clear()
                tcp_items = []
                wp_ori = [math.radians(v) for v in wp["tcp_ori_deg"]]
                moved = sim.move_to(wp["tcp_pos"], wp_ori, linear=True, speed=current_speed)
                if not moved:
                    moved = sim.move_to(wp["tcp_pos"], wp_ori, linear=False, speed=current_speed)
                if not moved:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
            tcp_items = draw_tcp()
            waypoint_items = draw_waypoints()
            continue

        pybullet.removeAllUserDebugItems()
        items.clear()
        tcp_items = []
        waypoint_items = draw_waypoints()

        if cmd.action == "error":
            continue
        if cmd.action == "incomplete":
            tcp_items = draw_tcp()
            if sim._last_target:
                tcp_pos, _ = sim.get_tcp_pose()
                displacement_mm = math.sqrt(sum((tcp_pos[i]-sim._last_target[0][i])**2 for i in range(3)))
                if displacement_mm > 0.01:
                    print(f"  ⚠ {displacement_mm*1000:.0f}mm manuell verschoben (@ zum Reset)")
            continue

        target_position = cmd.params["target_position"]
        target_orientation = cmd.params["target_orientation"]
        linear = cmd.params["linear"]

        tcp_pos, _ = sim.get_tcp_pose()

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fds = (os.dup(1), os.dup(2))
        os.dup2(devnull_fd, 1), os.dup2(devnull_fd, 2)

        _draw_crosshair(target_position, [1, 1, 0], items,
                        f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

        sim._last_collision_tcp_pose = None
        sim._last_collision_conf = None
        sim._remove_ghost()

        probe_result = sim._probe_path(target_position, target_orientation, rrt=not linear)

        os.dup2(saved_fds[0], 1), os.dup2(saved_fds[1], 2)
        os.close(devnull_fd)

        last_valid_position, collision_position, deviation_mm, actual_endpoint, rrt_waypoints = probe_result

        target_pos, target_ori = draw_probe_preview(
            last_valid_position, collision_position, deviation_mm,
            rrt_waypoints, tcp_pos, target_position, target_orientation,
            linear,
        )

        if collision_position is not None:
            reachable_distance = math.sqrt(sum((last_valid_position[i]-tcp_pos[i])**2 for i in range(3)))
            target_distance = math.sqrt(sum((target_position[i]-tcp_pos[i])**2 for i in range(3)))
            if reachable_distance > 0.005 and target_distance > 0:
                sim._show_ghost()
                fraction = reachable_distance / target_distance
                _, current_orientation = sim.get_tcp_pose()
                mid_orientation = pybullet.getQuaternionSlerp(
                    current_orientation, pybullet.getQuaternionFromEuler(target_orientation), fraction,
                )
                try:
                    confirm = input("  Zum Ghost-Punkt bewegen? [Y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if confirm in ("", "y", "yes"):
                    ok = sim.move_to(last_valid_position, list(pybullet.getEulerFromQuaternion(mid_orientation)), linear=linear, speed=current_speed, target_conf=sim._last_collision_conf)
                else:
                    ok = False
            else:
                print("  Ziel von aktueller Position aus unerreichbar (kein gültiger Zwischenpunkt)")
                ok = False
        else:
            try:
                confirm = input("  Ausführen? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            ok = False
            if confirm in ("", "y", "yes"):
                ok = sim.move_to(target_pos, target_ori, linear=linear, speed=current_speed)

        sim._remove_ghost()
        pybullet.removeAllUserDebugItems()
        items.clear()
        tcp_items = draw_tcp()
        waypoint_items = draw_waypoints()

    if sim._mirror is not None:
        sim._mirror.close()

if __name__ == "__main__":
    demo_simulation()
