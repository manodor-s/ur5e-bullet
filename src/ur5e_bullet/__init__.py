import os
import math
import pybullet
from collections import namedtuple

from .sim import (
    UR5Sim,
    ROBOT_URDF_PATH,
    GEBISS_SCALE,
    _JAWS_DIR,
)

import importlib.util as _ilu
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.join(_pkg_dir, "..", "..")
_cfg_spec = _ilu.spec_from_file_location("config", os.path.join(_proj_root, "config.py"))
_cfg_mod = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
START_POSITIONS = _cfg_mod.START_POSITIONS


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


def demo_simulation():
    sim = UR5Sim()

    def draw_tcp():
        pos, _ = sim.get_tcp_pose()
        return _draw_crosshair(pos, [0, 1, 0], [])

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
            continue
        if cmd.action == "reset" and sim._last_conf is not None:
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            sim._execute([sim._last_conf])
            tcp_items = draw_tcp()
            continue
        if cmd.action == "speed":
            current_speed = cmd.params["speed"]
            print(f"  Geschwindigkeit: {current_speed:.2f}")
            pybullet.removeAllUserDebugItems()
            items.clear()
            tcp_items = []
            tcp_items = draw_tcp()
            continue
        if cmd.action == "render":
            sim._mirror._render_done.clear()
            sim._mirror.send_message({"render": True})
            print("  Render gestartet...")
            sim._mirror._render_done.wait(timeout=300)
            continue
        if cmd.action == "jaw":
            folder = cmd.params["folder"]
            jaw_type = cmd.params["type"]
            try:
                ok = sim.load_jaw(folder, jaw_type)
            except FileNotFoundError as e:
                print(f"  ! {e}")
                continue
            if not ok:
                continue
            print(f"  PyBullet: gebiss_{jaw_type} aus Ordner {folder}")
            if sim._mirror is not None:
                sim._mirror._jaw_done.clear()
                sim._mirror.send_message({"replace_jaw": {"folder": folder, "type": jaw_type}})
                sim._mirror._jaw_done.wait(timeout=10)
            tcp_items = draw_tcp()
            continue
        if cmd.action == "start_pos":
            name = cmd.params["name"]
            cfg = START_POSITIONS[name]
            print(f"  → jaw entfernt...")
            sim.unload_jaw()
            print(f"  → fahre zu {name}-Start...")
            moved = sim.move_to(cfg["tcp_pos"], cfg["tcp_ori"], linear=True, speed=current_speed)
            if not moved:
                moved = sim.move_to(cfg["tcp_pos"], cfg["tcp_ori"], linear=False, speed=current_speed)
            jpos = cfg["jaw_pos"]
            jeuler = cfg["jaw_euler"]
            sim.load_jaw_at(cfg["jaw_folder"], cfg["jaw_type"], jpos, jeuler)
            if sim._mirror is not None:
                sim._mirror.send_current()
            print(f"  → jaw eingefuegt ({cfg['jaw_type']}, pos=({jpos[0]:.3f}, {jpos[1]:.3f}, {jpos[2]:.3f}))")
            current_start = name
            waypoint_idx = 0
            wps = cfg["waypoints"]
            print(f"  Waypoints: {len(wps)}")
            for i, wp in enumerate(wps):
                print(f"    {i+1}: {wp.get('label', str(i+1))} ({wp['tcp_pos'][0]:.3f}, {wp['tcp_pos'][1]:.3f}, {wp['tcp_pos'][2]:.3f})")
            tcp_items = draw_tcp()
            continue
        if cmd.action == "waypoint_next":
            if current_start is None:
                print(f"  ? Keine Startposition aktiv (zuerst 'start <name>')")
                continue
            cfg = START_POSITIONS[current_start]
            wps = cfg["waypoints"]
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
                moved = sim.move_to(wp["tcp_pos"], wp["tcp_ori"], linear=True, speed=current_speed)
                if not moved:
                    moved = sim.move_to(wp["tcp_pos"], wp["tcp_ori"], linear=False, speed=current_speed)
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
            wps = cfg["waypoints"]
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
                moved = sim.move_to(wp["tcp_pos"], wp["tcp_ori"], linear=True, speed=current_speed)
                if not moved:
                    moved = sim.move_to(wp["tcp_pos"], wp["tcp_ori"], linear=False, speed=current_speed)
                if not moved:
                    print(f"  ⛔ {current_start} {lbl} nicht erreichbar")
            tcp_items = draw_tcp()
            continue

        pybullet.removeAllUserDebugItems()
        items.clear()
        tcp_items = []

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

    if sim._mirror is not None:
        sim._mirror.close()

if __name__ == "__main__":
    demo_simulation()
